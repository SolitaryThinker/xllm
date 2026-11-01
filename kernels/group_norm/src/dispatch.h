// The dispatcher: decides, for one call of one of the eight group-norm operations, which implementation runs it.
//
// Several kernels serve different shape classes (channels per group, dtype, affine, memory efficient, row count).
// Each registers here once (a registered implementation is called a node), from its own .cu, with `handles()` saying
// which shapes it takes and a priority.
// `forward()` / `backward()` route the call to the highest-priority registered implementation that handles the shape,
// and fall back to xllm's own kernels when none does or all decline. Those kernels are correct for every shape xllm can
// call, so the fallback is always available and no implementation can shadow another by link order.
//
// The environment variable GN_IMPL, read once, overrides the route: GN_IMPL=off sends every call to xllm's kernels,
// and GN_IMPL=fwd:name,bwd:name restricts the candidates to the implementation of that name in each direction.
//
// Registering, in the implementation's own .cu file:
//
//   #include "dispatch.h"
//   namespace {
//   bool my_handles(const gn::Shape& s) { return s.mode == gn::kFwd && s.cpg % 8 == 0 && s.cpg <= 128; }
//   bool my_run(const gn::Shape& s, const gn::Call& c) {
//     if (!my_checks(c)) return false;  // decline: the next node is tried, and xllm's kernels last
//     /* launch the kernel */
//     return true;
//   }
//   const bool my_registered = gn::register_fwd("my_name", /*prio=*/10, my_handles, my_run);
//   }  // namespace
//
// Nothing else needs editing: binding.cpp calls gn::forward / gn::backward only.
#pragma once

#include <torch/torch.h>

#include <cstdint>
#include <string>

namespace gn {

constexpr int64_t kFwd = 0;
constexpr int64_t kBwd = 1;
constexpr int64_t kRMS = 0;
constexpr int64_t kLayer = 1;

// What a handler may branch on. Everything is derived from the tensors run() was given.
struct Shape {
  int64_t mode = 0;             // kFwd or kBwd
  int64_t norm = 0;             // kRMS or kLayer
  int64_t groups = 0;           // G
  int64_t channels = 0;         // C = G * cpg
  int64_t cpg = 0;              // channels per group
  int64_t rows = 0;             // B * S
  at::ScalarType dtype = at::kFloat;  // compare with at::kBFloat16 / at::kHalf / at::kFloat
  bool affine = false;          // gamma.numel() > 0
  bool has_beta = false;        // beta.numel() > 0 (layer norm with affine)
  bool memory_efficient = false;  // backward only: x_or_y is y
};

// The tensors of one call, in run()'s order. Nullable pointers name the arguments this call does not take.
struct Call {
  const at::Tensor* x_or_y = nullptr;
  const at::Tensor* dy = nullptr;
  const at::Tensor* mean = nullptr;
  const at::Tensor* rstd = nullptr;
  const at::Tensor* gamma = nullptr;
  const at::Tensor* beta = nullptr;
  at::Tensor* y = nullptr;
  at::Tensor* mean_out = nullptr;
  at::Tensor* rstd_out = nullptr;
  at::Tensor* dx = nullptr;
  at::Tensor* dgamma = nullptr;
  at::Tensor* dbeta = nullptr;
  double eps = 0.0;
};

using HandleFn = bool (*)(const Shape&);
// Returns true when it ran the call. False declines it, having launched nothing: the dispatcher then tries the next
// node in priority order, and xllm's kernels last. Declining is how a node whose claim is expressed on the Shape alone
// (a mode, a width) says "not this call after all" (a misaligned tensor, say) without denying the shapes below it.
using RunFn = bool (*)(const Shape&, const Call&);

// Returns false when the registry is off (GN_IMPL=off); a handler may ignore it.
bool register_fwd(const char* name, int prio, HandleFn handles, RunFn run);
bool register_bwd(const char* name, int prio, HandleFn handles, RunFn run);

// The two entry points binding.cpp uses. Each tries the nodes that claim the shape from the highest priority down
// (ties by name) until one runs the call, then falls back to xllm's kernels.
void forward(const Shape& s, const Call& c);
void backward(const Shape& s, const Call& c);

// xllm's own kernels (group_rms_norm_kernel.cu, group_layer_norm_kernel.cu): correct for every shape.
void xllm_forward(const Shape& s, const Call& c);
void xllm_backward(const Shape& s, const Call& c);

}  // namespace gn
