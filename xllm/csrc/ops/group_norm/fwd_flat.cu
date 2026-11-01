// Group RMS / layer norm forward pass, many groups, streaming.
//
// The tensor x[rows, groups, cpg] is one contiguous array of rows*groups groups, and rstd/mean are
// [rows, groups], so a group is a contiguous `cpg`-channel segment and its statistics index is the
// group's flat index. No (row, group) grid is needed: one grid-stride loop over groups.
//
// A group is handled by TPG consecutive lanes of a warp, Q 16-byte chunks each:
//     TPG = next_pow2(ceil(chunks / Q)),   chunks = cpg / VEC,   VEC = 16 / sizeof(T)
// so a group of 24, 40, 64, 80 or 128 channels is 3, 5, 8, 10 or 16 lanes; a group of 1024..4096
// channels is one or more whole warps. The statistics come from warp shuffles (a shared-memory
// combine of warp partials only when TPG > 32), and every value stays in registers between the
// reduction and the scale, so x is read once and y written once.
//
// The layer norm takes the mean first and the variance from the centered values, so rstd matches the
// two-pass definition to rounding rather than to the looser cancellation bound of a one-pass formula.
#include "fwd_flat.h"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>

namespace gn {
namespace {

constexpr int kBlock = 256;
constexpr int kWarps = kBlock / 32;

__device__ __forceinline__ float cvt(__nv_bfloat16 v) { return __bfloat162float(v); }
__device__ __forceinline__ float cvt(__half v) { return __half2float(v); }
__device__ __forceinline__ float cvt(float v) { return v; }
__device__ __forceinline__ void put(__nv_bfloat16& d, float s) { d = __float2bfloat16(s); }
__device__ __forceinline__ void put(__half& d, float s) { d = __float2half(s); }
__device__ __forceinline__ void put(float& d, float s) { d = s; }

template <typename T, int VEC>
struct alignas(16) Chunk {
  T e[VEC];
};

// Sum over the TPG lanes of a group, broadcast to all of them. TPG <= 32: shuffles only.
template <int TPG>
__device__ __forceinline__ float warp_sum_bcast(float v) {
  if constexpr (TPG > 1) {
#pragma unroll
    for (int d = TPG / 2; d > 0; d >>= 1) v += __shfl_down_sync(0xffffffffu, v, d);
    v = __shfl_sync(0xffffffffu, v, (int)(threadIdx.x & 31) & ~(TPG - 1));
  }
  return v;
}

// TPG > 32: whole warps per group, so the warp partials are combined through shared memory.
// Two barriers per call: one before the read of the partials, one before the next write of them.
template <int TPG>
__device__ __forceinline__ float shm_sum_bcast(float v, float* shm, int slot) {
  constexpr int WPG = TPG / 32;
#pragma unroll
  for (int d = 16; d > 0; d >>= 1) v += __shfl_down_sync(0xffffffffu, v, d);
  if ((threadIdx.x & 31) == 0) shm[slot * kWarps + (threadIdx.x >> 5)] = v;
  __syncthreads();
  const int base = (int)(threadIdx.x / TPG) * WPG;
  float t = 0.f;
#pragma unroll
  for (int i = 0; i < WPG; ++i) t += shm[slot * kWarps + base + i];
  __syncthreads();
  return t;
}

template <typename T, int TPG, int Q, bool LAYER, bool AFFINE>
__global__ __launch_bounds__(kBlock) void fwd_kernel(const T* __restrict__ x, const T* __restrict__ gamma,
                                                     const T* __restrict__ beta, T* __restrict__ yout,
                                                     float* __restrict__ mean_out, float* __restrict__ rstd,
                                                     unsigned n_groups, unsigned groups, int chunks, int cpg,
                                                     float eps, float inv_c) {
  constexpr int VEC = 16 / (int)sizeof(T);
  constexpr int GPB = kBlock / TPG;  // groups per CTA iteration
  __shared__ float shm[2 * kWarps];
  const int tid = (int)threadIdx.x;
  const int p = tid & (TPG - 1);   // chunk slot inside the group
  const int gsub = tid / TPG;      // group slot inside the CTA
  const unsigned step = (unsigned)gridDim.x * GPB;

  for (unsigned g0 = (unsigned)blockIdx.x * GPB; g0 < n_groups; g0 += step) {
    if constexpr (TPG > 32) __syncthreads();
    const unsigned g = g0 + (unsigned)gsub;
    const bool active = g < n_groups;
    const size_t goff = (size_t)g * (size_t)cpg;
    const T* xg = x + goff;

    Chunk<T, VEC> v[Q];
    // The layer norm would convert each element three times -- once for the sum, once for the centered
    // sum of squares, once for the output -- so it keeps the converted values instead. This is selected
    // on LAYER at compile time: the RMS norm has only one pass to feed, and the kept values would cost
    // registers there and buy nothing.
    float fv[Q][VEC];
    float s = 0.f;   // sum of squares (RMS: the only pass) or, for the layer norm, unused
    float sx = 0.f;  // sum, layer norm only
#pragma unroll
    for (int i = 0; i < Q; ++i) {
      const int chunk_idx = p + i * TPG;
      if (active && chunk_idx < chunks) {
        v[i] = *reinterpret_cast<const Chunk<T, VEC>*>(xg + (size_t)chunk_idx * VEC);
#pragma unroll
        for (int j = 0; j < VEC; ++j) {
          const float f = cvt(v[i].e[j]);
          if constexpr (LAYER) {
            fv[i][j] = f;
            sx += f;
          } else {
            s = fmaf(f, f, s);
          }
        }
      }
    }

    float mu = 0.f;
    float var;
    if constexpr (LAYER) {
      mu = ((TPG > 32) ? shm_sum_bcast<TPG>(sx, shm, 0) : warp_sum_bcast<TPG>(sx)) * inv_c;
      float c2 = 0.f;
#pragma unroll
      for (int i = 0; i < Q; ++i) {
        const int chunk_idx = p + i * TPG;
        if (active && chunk_idx < chunks) {
#pragma unroll
          for (int j = 0; j < VEC; ++j) {
            const float d = fv[i][j] - mu;
            c2 = fmaf(d, d, c2);
          }
        }
      }
      var = ((TPG > 32) ? shm_sum_bcast<TPG>(c2, shm, 1) : warp_sum_bcast<TPG>(c2)) * inv_c;
    } else {
      var = ((TPG > 32) ? shm_sum_bcast<TPG>(s, shm, 0) : warp_sum_bcast<TPG>(s)) * inv_c;
    }

    const float r = rsqrtf(var + eps);
    if (active && p == 0) {
      rstd[g] = r;
      if constexpr (LAYER) mean_out[g] = mu;
    }

    const float a = r;
    const float b = -mu * r;
    const size_t coff = AFFINE ? (size_t)(g % groups) * (size_t)cpg : (size_t)0;
    T* yg = yout + goff;
#pragma unroll
    for (int i = 0; i < Q; ++i) {
      const int chunk_idx = p + i * TPG;
      if (active && chunk_idx < chunks) {
        const size_t off = (size_t)chunk_idx * VEC;
        Chunk<T, VEC> o;
        if constexpr (AFFINE) {
          const Chunk<T, VEC> gv = *reinterpret_cast<const Chunk<T, VEC>*>(gamma + coff + off);
          if constexpr (LAYER) {
            // `beta` is read on every affine LAYER instantiation, so a layer norm with gamma must come with a
            // real beta pointer. xllm passes gamma and beta together, and its own kernel makes the same
            // assumption. A layer norm with gamma and an empty beta would fault here.
            const Chunk<T, VEC> bv = *reinterpret_cast<const Chunk<T, VEC>*>(beta + coff + off);
#pragma unroll
            for (int j = 0; j < VEC; ++j) put(o.e[j], (fv[i][j] * a + b) * cvt(gv.e[j]) + cvt(bv.e[j]));
          } else {
#pragma unroll
            for (int j = 0; j < VEC; ++j) put(o.e[j], (cvt(v[i].e[j]) * a + b) * cvt(gv.e[j]));
          }
        } else {
          if constexpr (LAYER) {
#pragma unroll
            for (int j = 0; j < VEC; ++j) put(o.e[j], fv[i][j] * a + b);
          } else {
#pragma unroll
            for (int j = 0; j < VEC; ++j) put(o.e[j], cvt(v[i].e[j]) * a + b);
          }
        }
        *reinterpret_cast<Chunk<T, VEC>*>(yg + off) = o;
      }
    }
  }
}

// Any channels per group: one warp per group, scalar, x read twice. Correct everywhere, used only
// where the vector width does not divide cpg (or where the flat index does not fit an unsigned).
template <typename T, bool LAYER, bool AFFINE>
__global__ __launch_bounds__(kBlock) void fwd_generic(const T* __restrict__ x, const T* __restrict__ gamma,
                                                      const T* __restrict__ beta, T* __restrict__ yout,
                                                      float* __restrict__ mean_out, float* __restrict__ rstd,
                                                      unsigned long long n_groups, unsigned long long groups,
                                                      int cpg, float eps, float inv_c) {
  const unsigned long long g = (unsigned long long)blockIdx.x * (kBlock / 32) + (threadIdx.x >> 5);
  if (g >= n_groups) return;
  const int lane = (int)(threadIdx.x & 31);
  const size_t goff = (size_t)g * (size_t)cpg;
  const T* xg = x + goff;

  float s = 0.f;
  float sx = 0.f;
  for (int i = lane; i < cpg; i += 32) {
    const float f = cvt(xg[i]);
    s = fmaf(f, f, s);
    if constexpr (LAYER) sx += f;
  }
#pragma unroll
  for (int d = 16; d > 0; d >>= 1) s += __shfl_xor_sync(0xffffffffu, s, d);
  if constexpr (LAYER) {
#pragma unroll
    for (int d = 16; d > 0; d >>= 1) sx += __shfl_xor_sync(0xffffffffu, sx, d);
  }

  float mu = 0.f;
  float var;
  if constexpr (LAYER) {
    mu = sx * inv_c;
    float c2 = 0.f;
    for (int i = lane; i < cpg; i += 32) {
      const float d = cvt(xg[i]) - mu;
      c2 = fmaf(d, d, c2);
    }
#pragma unroll
    for (int d = 16; d > 0; d >>= 1) c2 += __shfl_xor_sync(0xffffffffu, c2, d);
    var = c2 * inv_c;
  } else {
    var = s * inv_c;
  }

  const float r = rsqrtf(var + eps);
  if (lane == 0) {
    rstd[g] = r;
    if constexpr (LAYER) mean_out[g] = mu;
  }

  const size_t coff = AFFINE ? (size_t)(g % groups) * (size_t)cpg : (size_t)0;
  T* yg = yout + goff;
  for (int i = lane; i < cpg; i += 32) {
    float v = (cvt(xg[i]) - mu) * r;
    if constexpr (AFFINE) {
      v *= cvt(gamma[coff + i]);
      if constexpr (LAYER) v += cvt(beta[coff + i]);
    }
    put(yg[i], v);
  }
}

int sm_count() {
  static int n = 0;
  if (n == 0) {
    int dev = 0;
    cudaGetDevice(&dev);
    if (cudaDeviceGetAttribute(&n, cudaDevAttrMultiProcessorCount, dev) != cudaSuccess || n <= 0) n = 132;
  }
  return n;
}

int grid_for(unsigned n_groups, int gpb, int bpsm) {
  const long long tiles = ((long long)n_groups + gpb - 1) / gpb;
  const long long want = (long long)sm_count() * bpsm;
  return (int)std::min(tiles, want);
}

template <typename T, int TPG, int Q, bool LAYER, bool AFFINE>
void launch_one(const FwdParams& p, int chunks, cudaStream_t stream) {
  const unsigned n = (unsigned)(p.rows * p.groups);
  const int grid = grid_for(n, kBlock / TPG, p.grid_bpsm);
  fwd_kernel<T, TPG, Q, LAYER, AFFINE><<<grid, kBlock, 0, stream>>>(
      (const T*)p.x, (const T*)p.gamma, (const T*)p.beta, (T*)p.y, (float*)p.mean, (float*)p.rstd, n, p.groups,
      chunks, (int)p.cpg, p.eps, 1.f / (float)p.cpg);
}

template <typename T, int TPG, int Q>
void launch_tq(const FwdParams& p, int chunks, cudaStream_t stream) {
  if (p.layer != 0) {
    if (p.gamma != nullptr) launch_one<T, TPG, Q, true, true>(p, chunks, stream);
    else launch_one<T, TPG, Q, true, false>(p, chunks, stream);
  } else {
    if (p.gamma != nullptr) launch_one<T, TPG, Q, false, true>(p, chunks, stream);
    else launch_one<T, TPG, Q, false, false>(p, chunks, stream);
  }
}

template <typename T, bool LAYER, bool AFFINE>
void launch_generic_one(const FwdParams& p, unsigned long long n, int grid, cudaStream_t stream) {
  fwd_generic<T, LAYER, AFFINE><<<grid, kBlock, 0, stream>>>(
      (const T*)p.x, (const T*)p.gamma, (const T*)p.beta, (T*)p.y, (float*)p.mean, (float*)p.rstd, n, p.groups,
      (int)p.cpg, p.eps, 1.f / (float)p.cpg);
}

template <typename T>
void launch_generic_t(const FwdParams& p, unsigned long long n, int grid, cudaStream_t stream) {
  if (p.layer != 0) {
    if (p.gamma != nullptr) launch_generic_one<T, true, true>(p, n, grid, stream);
    else launch_generic_one<T, true, false>(p, n, grid, stream);
  } else {
    if (p.gamma != nullptr) launch_generic_one<T, false, true>(p, n, grid, stream);
    else launch_generic_one<T, false, false>(p, n, grid, stream);
  }
}

// One warp a group, the group read twice (this is the fallback for groups that are not a whole number
// of 16-byte chunks). The grid covers every group: no cap, because the kernel has no stride loop.
void launch_generic(const FwdParams& p, cudaStream_t stream) {
  const unsigned long long n = p.rows * p.groups;
  const long long tiles = ((long long)n + (kBlock / 32) - 1) / (kBlock / 32);
  const int grid = (int)std::min<long long>(tiles, 0x7fffffffLL);
  if (p.dtype == 0) launch_generic_t<__nv_bfloat16>(p, n, grid, stream);
  else if (p.dtype == 1) launch_generic_t<__half>(p, n, grid, stream);
  else launch_generic_t<float>(p, n, grid, stream);
}

}  // namespace

namespace {

// HALF=false: the group spread over enough lanes that one lane loads one chunk (Q=1 for every cpg
// up to 256); HALF=true: half as many lanes a group, so a warp holds twice as many whole groups
// (8 instead of 4 at cpg 64/128) and a lane keeps 2-4 chunks in flight.
template <bool HALF>
void fwd_dispatch(const FwdParams& p, cudaStream_t st) {
  const int esz = (p.dtype == 2) ? 4 : 2;
  const int vec = 16 / esz;
  const unsigned long long n = p.rows * p.groups;
  const int cpg = (int)p.cpg;
  const int chunks = cpg / vec;
  const bool vectorizable = (cpg % vec == 0) && chunks >= 1 && chunks <= 1024 && n < 0xffffffffull;

  if (!vectorizable) {
    launch_generic(p, st);
    return;
  }

#define GN_FWD_CASE(T, TPG, Q)                                    \
  else if (chunks <= (TPG) * (Q)) {                               \
    launch_tq<T, TPG, Q>(p, chunks, st);                          \
    return;                                                       \
  }

#define GN_FWD_HEAD(T)                                            \
  if (chunks <= 1) {                                              \
    launch_tq<T, 1, 1>(p, chunks, st);                            \
    return;                                                       \
  }                                                               \
  GN_FWD_CASE(T, 2, 1)

// The arm that takes 257..512 chunks is (128, 4): four warps a group, 4 chunks (64 B) of data a lane. It serves every
// 512-chunk group, whatever its norm or dtype (cpg 4096 in 16 bits, cpg 2048 in fp32). 64 B a lane reads faster
// than 128 B a lane, as the narrower groups of this chain do. (32, 16), the alternative that keeps the reduction
// inside one warp, is 3.5x slower: at this width the constraint is the data registers a lane, not the width of the
// reduction.
#define GN_FWD_CHAIN(T)                                           \
  GN_FWD_HEAD(T)                                                  \
  GN_FWD_CASE(T, 4, 1)                                            \
  GN_FWD_CASE(T, 8, 1)                                            \
  GN_FWD_CASE(T, 16, 1)                                           \
  GN_FWD_CASE(T, 32, 1)                                           \
  GN_FWD_CASE(T, 32, 2)                                           \
  GN_FWD_CASE(T, 32, 4)                                           \
  GN_FWD_CASE(T, 32, 5)                                           \
  GN_FWD_CASE(T, 32, 6)                                           \
  GN_FWD_CASE(T, 32, 8)                                           \
  GN_FWD_CASE(T, 128, 4)                                          \
  GN_FWD_CASE(T, 128, 8)

// The same kernel with half as many lanes a group, so a warp holds twice as many whole groups (8 at
// cpg 40, 64, 80 and 128 instead of 4) and a lane keeps 2-4 chunks in flight. Only the shapes with
// up to 32 chunks a group change: at 64 chunks and above, whole warps a group and the shortest
// register path win, so that half of the chain is the one above.
#define GN_FWD_CHAIN_HALF(T)                                      \
  GN_FWD_HEAD(T)                                                  \
  GN_FWD_CASE(T, 2, 2)                                            \
  GN_FWD_CASE(T, 4, 2)                                            \
  GN_FWD_CASE(T, 4, 4)                                            \
  GN_FWD_CASE(T, 8, 4)                                            \
  GN_FWD_CASE(T, 32, 2)                                           \
  GN_FWD_CASE(T, 32, 4)                                           \
  GN_FWD_CASE(T, 32, 5)                                           \
  GN_FWD_CASE(T, 32, 6)                                           \
  GN_FWD_CASE(T, 32, 8)                                           \
  GN_FWD_CASE(T, 64, 8)                                           \
  GN_FWD_CASE(T, 128, 8)

  if (p.dtype == 0) {
    if constexpr (HALF) {
      GN_FWD_CHAIN_HALF(__nv_bfloat16)
    } else {
      GN_FWD_CHAIN(__nv_bfloat16)
    }
  } else if (p.dtype == 1) {
    if constexpr (HALF) {
      GN_FWD_CHAIN_HALF(__half)
    } else {
      GN_FWD_CHAIN(__half)
    }
  } else {
    if constexpr (HALF) {
      GN_FWD_CHAIN_HALF(float)
    } else {
      GN_FWD_CHAIN(float)
    }
  }
#undef GN_FWD_CHAIN_HALF
#undef GN_FWD_CHAIN
#undef GN_FWD_HEAD
#undef GN_FWD_CASE
  launch_generic(p, st);
}

}  // namespace

void fwd_launch(const FwdParams& p, void* stream) { fwd_dispatch<false>(p, (cudaStream_t)stream); }

void fwd_launch_half(const FwdParams& p, void* stream) { fwd_dispatch<true>(p, (cudaStream_t)stream); }

}  // namespace gn

// ---------------------------------------------------------------------------------------------------
// The dispatch node (see dispatch.h): claims every forward shape that is not excluded below, and declines the calls
// whose tensors it cannot read, so the next node in the order -- or xllm's kernels -- takes them.
#include <c10/cuda/CUDAStream.h>

#include "dispatch.h"

namespace {

bool fwd_flat_aligned(const at::Tensor& t) {
  return t.is_contiguous() && (reinterpret_cast<uintptr_t>(t.data_ptr()) & 15u) == 0;
}

int64_t fwd_flat_chunks(const gn::Shape& s) {
  const int64_t esz = (s.dtype == at::kFloat) ? 4 : 2;
  return s.cpg / (16 / esz);
}

bool fwd_flat_handles(const gn::Shape& s) {
  if (s.mode != gn::kFwd || s.cpg < 1 || s.rows < 1 || s.groups < 1) return false;
  if (s.dtype != at::kBFloat16 && s.dtype != at::kHalf && s.dtype != at::kFloat) return false;
  const int64_t esz = (s.dtype == at::kFloat) ? 4 : 2;
  const int64_t vec = 16 / esz;
  // The vectorized path wants a whole number of 16-byte chunks a group; the one-warp-a-group fallback
  // (fwd_generic) takes any group width, so the node covers every forward shape xllm can call.
  const int64_t chunks = s.cpg / vec;
  if (s.cpg % vec == 0 && chunks > 1024) return false;
  // A group of 3 or 5 chunks (cpg 24 and 40 in 16 bits) fills only a third to a fifth of a warp's 16-byte slots when
  // mapped to a power of two of lanes; the dense-lane kernels map such a group to exactly that many lanes, so they
  // take these widths.
  if (s.cpg % vec == 0 && (chunks == 3 || chunks == 5)) return false;
  return (unsigned long long)s.rows * (unsigned long long)s.groups < 0xffffffffull;
}

// The grid length, per shape: how many CTAs of 256 threads per SM the grid holds (each walks a grid-stride slice of
// the tiles). 64 for 8 and 10 chunks (cpg 64 and 80 in 16 bits), for the layer norm, and for 512 chunks and wider;
// 128 for everything else.
int fwd_flat_bpsm(int64_t chunks, int64_t norm) {
  if (chunks >= 512 || norm == gn::kLayer) return 64;
  if (chunks == 5 || chunks == 8 || chunks == 10) return 64;
  return 128;
}

// half: half as many lanes a group (a warp holds 8 whole groups at cpg 64 and 128). Groups of 512
// chunks and wider keep the flat mapping: there a whole warp a group and the shortest register path win.
bool fwd_flat_launch(const gn::Shape& s, const gn::Call& c, int bpsm, bool half) {
  const at::Tensor& x = *c.x_or_y;
  if (!fwd_flat_aligned(x) || !fwd_flat_aligned(*c.y) || (s.affine && !fwd_flat_aligned(*c.gamma)) ||
      (s.has_beta && !fwd_flat_aligned(*c.beta))) {
    return false;
  }
  gn::FwdParams p{};
  p.x = x.data_ptr();
  p.gamma = s.affine ? c.gamma->data_ptr() : nullptr;
  p.beta = s.has_beta ? c.beta->data_ptr() : nullptr;
  p.y = c.y->data_ptr();
  p.mean = (s.norm == gn::kLayer) ? c.mean_out->data_ptr() : nullptr;
  p.rstd = c.rstd_out->data_ptr();
  p.rows = (unsigned long long)s.rows;
  p.groups = (unsigned)s.groups;
  p.cpg = (unsigned)s.cpg;
  p.eps = (float)c.eps;
  p.dtype = s.dtype == at::kBFloat16 ? 0 : (s.dtype == at::kHalf ? 1 : 2);
  p.layer = s.norm == gn::kLayer ? 1 : 0;
  p.grid_bpsm = bpsm;
  void* st = (void*)c10::cuda::getCurrentCUDAStream().stream();
  if (half) gn::fwd_launch_half(p, st);
  else gn::fwd_launch(p, st);
  return true;
}

// Runs the vectorized kernel for a group of 1..1024 chunks, or the scalar one-warp-a-group kernel for a width that is
// not a whole number of chunks. Groups of 512 chunks and wider keep the flat lane mapping (a whole warp a group, one
// chunk a lane); narrower ones use the half-lane mapping.
bool fwd_flat_run(const gn::Shape& s, const gn::Call& c) {
  const int64_t chunks = fwd_flat_chunks(s);
  // Groups of 512 chunks and wider keep the flat lane mapping (a whole warp a group, one chunk a lane).
  return fwd_flat_launch(s, c, fwd_flat_bpsm(chunks, s.norm), chunks < 512);
}

const bool fwd_flat_registered =
    gn::register_fwd("fwd_flat", /*prio=*/25, fwd_flat_handles, fwd_flat_run);

}  // namespace
