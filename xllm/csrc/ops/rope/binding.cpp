// The entry point: one call of xllm's rotary position embedding. Called positionally with the inputs, then the
// outputs (destination passing): run(conj, x1, x2, freqs_cis, out1, out2).
//
//   conj  false: the forward pass; true: the backward pass (the rotation by the conjugate)
//   x1    [B, L, H, R], any strides with the last one 1 (a view of a wider head, or of q and kq side by side)
//   x2    the second tensor of apply_rotary_embeddings (k, [B, L, Hk, R]), or an empty tensor
//   freqs_cis  [L, R / 2] complex64, a slice of the table (it may start at an offset in its storage)
//
// Nothing here does ATen work: the shapes and strides are read and handed to one kernel (rope.cu).
#include <torch/extension.h>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>

#include "rope.h"

namespace {

RopeTens describe(const at::Tensor& x, const at::Tensor& out) {
  RopeTens t{};
  t.in = x.data_ptr();
  t.out = out.data_ptr();
  t.heads = static_cast<int>(x.size(2));
  t.sB = x.stride(0);
  t.sL = x.stride(1);
  t.sH = x.stride(2);
  t.oB = out.stride(0);
  t.oL = out.stride(1);
  t.oH = out.stride(2);
  return t;
}

}  // namespace

void RotaryEmbeddingCUDA(bool conj, const at::Tensor& x1, const at::Tensor& x2, const at::Tensor& freqs_cis, at::Tensor out1,
         at::Tensor out2) {
  // The kernel runs on the stream of x's device, as ATen ops do: a multi-GPU host must not have it run on whatever
  // device happens to be current.
  const c10::cuda::CUDAGuard guard(x1.device());
  TORCH_CHECK(x1.is_cuda() && x1.dim() == 4, "x1: [B, L, H, R] on the GPU");
  TORCH_CHECK(x1.stride(3) == 1, "x1: the last stride is 1");
  const auto dt = x1.scalar_type();
  TORCH_CHECK(dt == at::kBFloat16 || dt == at::kHalf || dt == at::kFloat, "x1: bf16, fp16 or fp32");
  TORCH_CHECK(x1.size(3) % 2 == 0, "x1: an even number of rotated channels (they are R / 2 complex pairs)");
  TORCH_CHECK(out1.is_contiguous() && out1.sizes() == x1.sizes() && out1.scalar_type() == dt, "out1: x1's shape");
  TORCH_CHECK(out1.device() == x1.device(), "out1: on x1's device");
  TORCH_CHECK(freqs_cis.scalar_type() == at::kComplexFloat && freqs_cis.dim() == 2, "freqs_cis: [L, R / 2] c64");
  TORCH_CHECK(freqs_cis.device() == x1.device(), "freqs_cis: on x1's device");
  // The launch's own limits, far outside what xllm rotates (32-256 channels, batches of a few), are checked here so
  // that an out-of-domain call is refused with a message rather than by an invalid launch.
  //
  // B is grid.y (65535). The positions live in grid.x (2^31 - 1 blocks, divided by the position tile), so L is bounded
  // only by the int the launch takes. R has no grid dimension; it is bounded below. The shaped launch covers a row of
  // up to 256 vectors of the widest tier that the width and the alignment admit (2048 channels in bf16, 1024 in fp32
  // when R is a multiple of that vector and every row is aligned; fewer when they are not, since a narrower tier means
  // more lanes); a wider row takes the wide path, which has no grid width limit. The head axis is a grid dimension too:
  // the row slots live in grid.z (65535) and the shaped body has no z stride, so a call whose row slots pass that hands
  // itself to the pair body, which strides over gridDim.z and whose z extent is capped. rope.cu documents the two
  // fallbacks and the row counts at which they start (67,107,840 rows at R = 2; 2,100,000 rows at R = 256 in bf16).
  TORCH_CHECK(x1.size(0) <= 65535, "x1: B up to 65535");
  TORCH_CHECK(x1.size(1) <= 2147483647, "x1: L up to 2^31 - 1 positions");
  // Heads are served up to the count the row total can hold: `rows` is an int sum of the two head counts in the
  // launchers and in the kernel, so a call whose heads together pass 2^31 - 1 would wrap negative, and a negative row
  // count is a launch that writes nothing and returns success: zeros handed to a caller as a gradient. It is refused
  // here with a message instead. Every host sum that scales the row count carries its numerator in `long`. Each
  // tensor's own head count is checked too, since `RopeTens::heads` is an int.
  TORCH_CHECK(x1.size(2) <= 2147483647, "x1: H up to 2^31 - 1 heads");
  const int64_t rows64 = x1.size(2) + (x2.dim() == 4 ? x2.size(2) : 0);
  TORCH_CHECK(rows64 <= 2147483647, "x1 and x2: the two tensors' heads together up to 2^31 - 1 (rows is an int)");
  // The width is carried in an `int` (the launcher takes `int R`), and above 2^31 - 1 it would fail silently in two
  // ways. `R = 2^31` casts to INT_MIN, so `rope_launch`'s opening test takes the empty-call path and returns success
  // without launching: the outputs keep whatever the caller allocated. `R = 2^32 + k` casts to `k`, rotates a k-channel
  // prefix and returns success on a row whose other channels are untouched, which reads as a working kernel. The bound
  // is therefore `<= 2^31 - 1` and not a test against the value that happens to cast negative: with the even test
  // above, exactly `R <= 2^31 - 2` is admitted, and the endpoint is served (one head, L = 1, bf16: out equals x bit
  // for bit on an identity table) through `launch_wide` with `zs = 1`, correct but not fast. Serving a larger R would
  // mean carrying the width in 64 bits through the launchers and the bodies.
  TORCH_CHECK(x1.size(3) <= 2147483647, "x1: R up to 2^31 - 1 rotated channels");
  TORCH_CHECK(freqs_cis.size(0) >= x1.size(1) && freqs_cis.size(1) == x1.size(3) / 2,
              "freqs_cis: [L, R / 2] with L at least x1's");

  const int dtype = dt == at::kBFloat16 ? 0 : (dt == at::kHalf ? 1 : 2);
  RopeTens t2{};
  if (x2.numel() > 0) {
    TORCH_CHECK(x2.is_cuda() && x2.dim() == 4 && x2.stride(3) == 1, "x2: [B, L, Hk, R] on the GPU");
    TORCH_CHECK(x2.device() == x1.device() && out2.device() == x1.device(), "x2, out2: on x1's device");
    TORCH_CHECK(x2.sizes() == out2.sizes() && out2.is_contiguous() && x2.scalar_type() == dt && out2.scalar_type() == dt, "out2: x2's shape");
    TORCH_CHECK(x2.size(0) == x1.size(0) && x2.size(1) == x1.size(1) && x2.size(3) == x1.size(3),
                "x2: B, L and R as x1's");
    t2 = describe(x2, out2);
  }
  const auto table = freqs_cis.resolve_conj().resolve_neg();
  rope_launch(conj, static_cast<int>(x1.size(0)), static_cast<int>(x1.size(1)), static_cast<int>(x1.size(3)),
              table.data_ptr(), table.stride(0), table.stride(1), dtype, describe(x1, out1), t2,
              at::cuda::getCurrentCUDAStream());
  // A launch the driver refuses must raise here: the outputs the caller allocated are zeroed, and a silent success
  // would hand a caller zeros as a gradient.
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

namespace xllm::ops {
void DefineRotaryEmbeddingOp(py::module& m) { m.def("rotary_embedding", &RotaryEmbeddingCUDA); }
}
