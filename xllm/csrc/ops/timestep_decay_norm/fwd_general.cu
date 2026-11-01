// The forward pass (mode 0/1) of xllm's group timestep decay normalization, for every shape xllm can call.
//
// One (row, group) chain's step is an affine map on each of four numbers (mean, var, count, and through the count
// the bias corrections), so a whole time tile of a chain is one affine aggregate, and the state a tile starts from is
// the composition of the aggregates of the earlier tiles of the same chain. That splits the work into
//   phase 0: each (chain, tile) reduces its steps to one 6-number aggregate (x read once),
//   phase 1: each (chain, tile) composes the earlier tiles' aggregates, scans its own steps and normalizes
//            (x read once, y written once).
// Both phases are the same kernel; with one tile per chain phase 0 is not needed and the kernel does both passes.
// Chains are (row, group) pairs; one warp walks one chain's tile, so any number of groups of any width works and the
// shuffles that reduce a group's channels stay inside one warp.
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

#include "fwd_general.h"

namespace xllm {
namespace ops {
namespace {

constexpr int kWarp = 32;
// Channels a lane holds of one step at most (register use bounds it): a group of up to 256 channels.
constexpr int kMaxNv = 8;

// ---------------------------------------------------------------------------------------------------------------
// Number formats: one conversion each way, so the kernels are written once for bf16, fp16 and fp32.

template <typename T>
struct Cvt;
template <>
struct Cvt<__nv_bfloat16> {
  static __device__ __forceinline__ float to_f(__nv_bfloat16 v) { return __bfloat162float(v); }
  static __device__ __forceinline__ __nv_bfloat16 from_f(float v) { return __float2bfloat16(v); }
  static __device__ __forceinline__ void unpack2(unsigned w, float* o) {
    const float2 f = __bfloat1622float2(*(const __nv_bfloat162*)&w);
    o[0] = f.x;
    o[1] = f.y;
  }
  static __device__ __forceinline__ unsigned pack2(float a, float b) {
    const __nv_bfloat162 h = __floats2bfloat162_rn(a, b);
    return *(const unsigned*)&h;
  }
};
template <>
struct Cvt<__half> {
  static __device__ __forceinline__ float to_f(__half v) { return __half2float(v); }
  static __device__ __forceinline__ __half from_f(float v) { return __float2half(v); }
  static __device__ __forceinline__ void unpack2(unsigned w, float* o) {
    const float2 f = __half22float2(*(const __half2*)&w);
    o[0] = f.x;
    o[1] = f.y;
  }
  static __device__ __forceinline__ unsigned pack2(float a, float b) {
    const __half2 h = __floats2half2_rn(a, b);
    return *(const unsigned*)&h;
  }
};
template <>
struct Cvt<float> {
  static __device__ __forceinline__ float to_f(float v) { return v; }
  static __device__ __forceinline__ float from_f(float v) { return v; }
  static __device__ __forceinline__ void unpack2(unsigned, float*) {}
  static __device__ __forceinline__ unsigned pack2(float, float) { return 0u; }
};

// ---------------------------------------------------------------------------------------------------------------
// The affine aggregate of a run of steps: v -> a * v + c, once for the running mean (1), the running variance (2) and
// the count (c). Applying "first" then "second" is what Compose(first, second) is.

struct __align__(32) Item {
  float a1, c1, a2, c2, ac, cc;
};

__device__ __forceinline__ Item ItemIdentity() { return Item{1.f, 0.f, 1.f, 0.f, 1.f, 0.f}; }

__device__ __forceinline__ Item ItemCompose(const Item& first, const Item& second) {
  Item r;
  r.a1 = second.a1 * first.a1;
  r.c1 = fmaf(second.a1, first.c1, second.c1);
  r.a2 = second.a2 * first.a2;
  r.c2 = fmaf(second.a2, first.c2, second.c2);
  r.ac = second.ac * first.ac;
  r.cc = fmaf(second.ac, first.cc, second.cc);
  return r;
}

// A valid step: mean <- beta1 * mean + (1 - beta1) * m_t (and 0 first if it is a bos step), the same for the
// variance with beta2, and count <- count + 1 (from 0 at a bos step). A padded step changes nothing.
__device__ __forceinline__ Item ItemStep(bool valid, bool reset, float m, float v, float b1, float b2, float om1,
                                         float om2) {
  if (!valid) return ItemIdentity();
  Item it;
  it.a1 = reset ? 0.f : b1;
  it.c1 = om1 * m;
  it.a2 = reset ? 0.f : b2;
  it.c2 = om2 * v;
  it.ac = reset ? 0.f : 1.f;
  it.cc = 1.f;
  return it;
}

// ---------------------------------------------------------------------------------------------------------------
// Sum of one value across the warp. Blackwell has no fp32 redux.sync.add (redux.sync.add is .u32/.s32 only, and f32
// only for min/max), so a butterfly of shuffles: 5 shuffles and 5 adds.
__device__ __forceinline__ float WarpSum(float v) {
#pragma unroll
  for (int off = 16; off > 0; off >>= 1) v += __shfl_xor_sync(0xffffffffu, v, off);
  return v;
}

// ---------------------------------------------------------------------------------------------------------------
// One step's group moments. `vec` says this lane's channels are contiguous (lane * NV .. + NV-1, and the group's
// 32 * NV channels are exactly the warp's), so a lane's 2, 4 or 8 channels are one 2-, 4-, 8- or 16-byte load and the
// warp's lanes are one contiguous run of the group: the fewest requests the hardware allows. Otherwise the group's
// channels are strided over the lanes (lane, lane + 32, ..), which is what an odd group width needs.
// The two passes over the values (sum, then the spread around the mean) make the variance as accurate as the
// reference's two-pass fp64 one at the tolerances asked for.
template <typename T, int NV>
__device__ __forceinline__ void LoadStep(const T* __restrict__ xp, int G, int lane, bool vec, float* xv, float& m,
                                         float& v) {
  const int nv = NV > 0 ? NV : (G + (kWarp - 1)) / kWarp;
  float s = 0.f;
  if (NV > 0 && vec) {
    const int base = lane * nv;
    if constexpr (sizeof(T) == 2) {
      if constexpr (NV == 1) {
        xv[0] = Cvt<T>::to_f(xp[base]);
        s = xv[0];
      } else {
        constexpr int NW = NV / 2;
        unsigned w[NW > 0 ? NW : 1];
        if constexpr (NW == 1) {
          w[0] = *(const unsigned*)(xp + base);
        } else if constexpr (NW == 2) {
          const uint2 t = *(const uint2*)(xp + base);
          w[0] = t.x;
          w[1] = t.y;
        } else if constexpr (NW == 4) {
          const uint4 t = *(const uint4*)(xp + base);
          w[0] = t.x;
          w[1] = t.y;
          w[2] = t.z;
          w[3] = t.w;
        }
#pragma unroll
        for (int i = 0; i < NW; ++i) Cvt<T>::unpack2(w[i], xv + 2 * i);
#pragma unroll
        for (int i = 0; i < NV; ++i) s += xv[i];
      }
    } else {
#pragma unroll
      for (int i = 0; i < NV; ++i) {
        xv[i] = xp[base + i];
        s += xv[i];
      }
    }
  } else {
#pragma unroll
    for (int i = 0; i < (NV > 0 ? NV : kMaxNv); ++i) {
      if (NV > 0 || i < nv) {
        const int c = lane + i * kWarp;
        const float t = (c < G) ? Cvt<T>::to_f(xp[c]) : 0.f;
        xv[i] = t;
        s += t;
      }
    }
  }
  const float inv_g = 1.f / (float)G;
  m = WarpSum(s) * inv_g;
  float ss = 0.f;
#pragma unroll
  for (int i = 0; i < (NV > 0 ? NV : kMaxNv); ++i) {
    if ((NV > 0 && vec) || NV > 0 || i < nv) {
      const int c = (NV > 0 && vec) ? 0 : lane + i * kWarp;
      if ((NV > 0 && vec) || c < G) {
        const float d = xv[i] - m;
        ss = fmaf(d, d, ss);
      }
    }
  }
  v = WarpSum(ss) * inv_g;
}

// The normalized step, in the same layout LoadStep read.
template <typename T, int NV>
__device__ __forceinline__ void StoreStep(T* __restrict__ yp, int G, int lane, bool vec, const float* xv, float cm,
                                          float cr, const float* gv, const float* bv) {
  const int nv = NV > 0 ? NV : (G + (kWarp - 1)) / kWarp;
  if (NV > 0 && vec) {
    const int base = lane * nv;
    if constexpr (sizeof(T) == 2) {
      float y[8];
#pragma unroll
      for (int i = 0; i < NV; ++i) {
        const float s = cr * gv[i];
        y[i] = fmaf(xv[i], s, fmaf(-cm, s, bv[i]));
      }
      if constexpr (NV == 1) {
        yp[base] = Cvt<T>::from_f(y[0]);
      } else {
        constexpr int NW = NV / 2;
        unsigned w[NW > 0 ? NW : 1];
#pragma unroll
        for (int i = 0; i < NW; ++i) w[i] = Cvt<T>::pack2(y[2 * i], y[2 * i + 1]);
        if constexpr (NW == 1) {
          *(unsigned*)(yp + base) = w[0];
        } else if constexpr (NW == 2) {
          *(uint2*)(yp + base) = make_uint2(w[0], w[1]);
        } else if constexpr (NW == 4) {
          *(uint4*)(yp + base) = make_uint4(w[0], w[1], w[2], w[3]);
        }
      }
    } else {
#pragma unroll
      for (int i = 0; i < NV; ++i) {
        const float s = cr * gv[i];
        yp[base + i] = fmaf(xv[i], s, fmaf(-cm, s, bv[i]));
      }
    }
  } else {
#pragma unroll
    for (int i = 0; i < (NV > 0 ? NV : kMaxNv); ++i) {
      if (NV > 0 || i < nv) {
        const int c = lane + i * kWarp;
        if (c < G) {
          const float s = cr * gv[i];
          yp[c] = Cvt<T>::from_f(fmaf(xv[i], s, fmaf(-cm, s, bv[i])));
        }
      }
    }
  }
}

// ---------------------------------------------------------------------------------------------------------------
// Phase 0: one warp reduces one (chain, tile) to its aggregate.
template <typename T, int NV, int WARPS>
__global__ __launch_bounds__(WARPS * kWarp) void TdnFwdAggKernel(const T* __restrict__ x,
                                                                const unsigned char* __restrict__ bos,
                                                                const unsigned char* __restrict__ pad,
                                                                Item* __restrict__ agg_out, int L, int H, int G, int K,
                                                                int tt, int ntiles, int chains, float b1, float b2) {
  const int lane = threadIdx.x & (kWarp - 1);
  const int warp = threadIdx.x >> 5;
  const int chain = blockIdx.x * WARPS + warp;
  if (chain >= chains) return;
  const long long b = chain / K;
  const int k = chain - (int)(b * K);
  const int t0 = blockIdx.y * tt;
  int t1 = t0 + tt;
  if (t1 > L) t1 = L;
  const unsigned char* bosp = bos ? bos + b * L : nullptr;
  const unsigned char* padp = pad ? pad + b * L : nullptr;
  const float om1 = 1.f - b1, om2 = 1.f - b2;
  const bool vec = NV > 0 && G == kWarp * NV;
  Item agg = ItemIdentity();
  float xv[kMaxNv];
  for (int t = t0; t < t1; ++t) {
    const bool valid = !(padp && padp[t]);
    const bool reset = valid && bosp && bosp[t];
    float m = 0.f, v = 0.f;
    if (valid) LoadStep<T, NV>(x + ((b * L + t) * H + (long long)k * G), G, lane, vec, xv, m, v);
    agg = ItemCompose(agg, ItemStep(valid, reset, m, v, b1, b2, om1, om2));
  }
  if (lane == 0) agg_out[(long long)chain * ntiles + blockIdx.y] = agg;
}

// ---------------------------------------------------------------------------------------------------------------
// Phase 1: one warp starts its chain's tile from the composed aggregates of the earlier tiles, scans its steps and
// normalizes them.
template <typename T, int NV, int WARPS>
__global__ __launch_bounds__(WARPS * kWarp) void TdnFwdNormKernel(
    const T* __restrict__ x, const unsigned char* __restrict__ bos, const unsigned char* __restrict__ pad,
    const int64_t* __restrict__ prev_count, const T* __restrict__ prev_mean, const T* __restrict__ prev_var,
    const T* __restrict__ gamma, const T* __restrict__ beta, T* __restrict__ y, int64_t* __restrict__ count_out,
    T* __restrict__ mean_out, T* __restrict__ var_out, float* __restrict__ cm_out, float* __restrict__ cr_out,
    const Item* __restrict__ agg_in, int L, int H, int G, int K, int tt, int ntiles, int chains,
    float om1, float om2, float eps, float dlog1, float dlog2) {
  const int lane = threadIdx.x & (kWarp - 1);
  const int warp = threadIdx.x >> 5;
  const int chain = blockIdx.x * WARPS + warp;
  if (chain >= chains) return;
  const int tile = blockIdx.y;
  const long long b = chain / K;
  const int k = chain - (int)(b * K);
  const int t0 = tile * tt;
  if (t0 >= L) return;
  int t1 = t0 + tt;
  if (t1 > L) t1 = L;

  // The state this tile starts from: the aggregates of tiles [0, tile) applied to the state carried in, and the count
  // they leave (a bos step inside them sets the count back to zero, which the aggregate's a = 0 carries).
  Item pre = ItemIdentity();
  if (tile > 0) {
    const Item* s = agg_in + (long long)chain * ntiles;
    for (int j = 0; j < tile; ++j) pre = ItemCompose(pre, s[j]);
  }
  const int nv = NV > 0 ? NV : (G + (kWarp - 1)) / kWarp;
  const bool vec = NV > 0 && G == kWarp * NV;
  const float pm = prev_mean ? Cvt<T>::to_f(prev_mean[b * K + k]) : 0.f;
  const float pv = prev_var ? Cvt<T>::to_f(prev_var[b * K + k]) : 0.f;
  const float cnt_prev = prev_count ? (float)prev_count[b] : 0.f;
  // The count the output reports is kept exactly, in int64: the fp32 state stops incrementing at 2^24 and count_out is
  // compared with atol = rtol = 0. Above 2^24 the fp32 count is only an argument to -expm1(log(beta) * count), where
  // the correction is already exactly 1 to fp32, so nothing else needs it exact.
  long long icnt = (pre.ac > 0.5f) ? (prev_count ? prev_count[b] : 0LL) + (long long)lrintf(pre.cc)
                                   : (long long)lrintf(pre.cc);
  float mean = fmaf(pre.a1, pm, pre.c1);
  float var = fmaf(pre.a2, pv, pre.c2);
  float cnt = fmaf(pre.ac, cnt_prev, pre.cc);

  // The per-channel scale and shift the normalization divides by, loaded once: this lane's channels are fixed.
  const long long hbase = (long long)k * G;
  float gv[kMaxNv], bv[kMaxNv];
#pragma unroll
  for (int i = 0; i < (NV > 0 ? NV : kMaxNv); ++i) {
    if (NV > 0 || i < nv) {
      const int c = vec ? lane * nv + i : lane + i * kWarp;
      if (vec || c < G) {
        gv[i] = Cvt<T>::to_f(gamma[hbase + c]);
        bv[i] = Cvt<T>::to_f(beta[hbase + c]);
      }
    }
  }

  const unsigned char* bosp = bos ? bos + b * L : nullptr;
  const unsigned char* padp = pad ? pad + b * L : nullptr;
  float* cmp = cm_out ? cm_out + b * L * K + k : nullptr;
  float* crp = cr_out ? cr_out + b * L * K + k : nullptr;
  const bool last = (t1 == L);

  float xv[kMaxNv];
  for (int t = t0; t < t1; ++t) {
    const bool valid = !(padp && padp[t]);
    T* yp = y + ((b * L + t) * H + hbase);
    if (valid) {
      const bool reset = bosp && bosp[t];
      float m = 0.f, v = 0.f;
      LoadStep<T, NV>(x + ((b * L + t) * H + hbase), G, lane, vec, xv, m, v);
      if (reset) {
        mean = 0.f;
        var = 0.f;
        cnt = 0.f;
        icnt = 0;
      }
      // The recurrence in delta form: `beta * mean + om * m` and `mean + om * (m - mean)` are the same map when
      // `om == 1 - beta`, but fp32 cannot hold that pair when the exact `1 - beta` is wanted: `1.f - (float)0.999`
      // carries the float beta's own rounding (1.3e-05 relative for beta1, 1.7e-04 for beta2). In delta form the
      // constant is a rate and the fixed point is `m` for any `om`, so `om` is the correctly rounded
      // `(float)(1.0 - beta)` computed on the host in double. One subtraction and one fma, the same two instructions
      // as the multiply-plus-fma form.
      // The tile aggregates (`ItemStep`) keep the weight form `a = beta, c = (1.f - beta) * m`, whose ratio
      // c / (1 - a) is `m` exactly.
      mean = fmaf(om1, m - mean, mean);
      var = fmaf(om2, v - var, var);
      cnt += 1.f;
      icnt += 1;
    }
    // A padded step keeps the state, and still reports the statistics it would normalize with (the reference's
    // cummean and cumrstd are defined at every step), while its y is zero.
    // The bias corrections come from the exact count every step: a running product would drift by the count's worth
    // of roundings, and cummean is asked for at 1e-3 relative.
    // Every lane of the warp carries the state: at these chain counts the kernel is bound by latency, not by issue,
    // so a shuffle on the critical path costs more than the instructions a single carrying lane would save.
    const float d1 = -expm1f(dlog1 * cnt);
    const float d2 = -expm1f(dlog2 * cnt);
    float cm, cr;
    if (d1 > 0.f && d2 > 0.f) {
      cm = mean / d1;
      cr = rsqrtf(var / d2 + eps);
    } else {
      cm = 0.f;
      cr = rsqrtf(eps);
    }
    if (lane == 0 && cmp) {
      cmp[t * K] = cm;
      crp[t * K] = cr;
    }
    if (valid) {
      StoreStep<T, NV>(yp, G, lane, vec, xv, cm, cr, gv, bv);
    } else {
#pragma unroll
      for (int i = 0; i < (NV > 0 ? NV : kMaxNv); ++i) {
        if (NV > 0 || i < nv) {
          const int c = vec ? lane * nv + i : lane + i * kWarp;
          if (vec || c < G) yp[c] = Cvt<T>::from_f(0.f);
        }
      }
    }
  }

  if (last && lane == 0) {
    if (mean_out) mean_out[b * K + k] = Cvt<T>::from_f(mean);
    if (var_out) var_out[b * K + k] = Cvt<T>::from_f(var);
    if (count_out && k == 0) count_out[b] = icnt;
  }
}

// ---------------------------------------------------------------------------------------------------------------
// Launch

struct Cfg {
  int warps;
  int ntiles;
  int tt;
};

Cfg ChooseCfg(long long chains, long long L) {
  const long long target_warps = 4096;  // warps the GPU can hold, roughly: tiles per chain that fill it
  long long ntiles = (target_warps + chains - 1) / chains;
  const long long shortest = L / 32;  // no tile shorter than 32 steps
  if (ntiles > shortest) ntiles = shortest;
  if (ntiles < 1) ntiles = 1;
  if (ntiles > 1024) ntiles = 1024;
  long long tt = (L + ntiles - 1) / ntiles;
  if (tt < 1) tt = 1;
  ntiles = (L + tt - 1) / tt;
  Cfg c;
  c.warps = 8;
  c.ntiles = (int)ntiles;
  c.tt = (int)tt;
  return c;
}

// Scratch for the aggregates, one Item per (chain, tile): allocated per call, so two forwards in flight at once never
// share it.
template <typename T, int NV>
void LaunchOne(const T* x, const unsigned char* bos, const unsigned char* pad, const int64_t* prev_count,
               const T* prev_mean, const T* prev_var, const T* gamma, const T* beta, T* y, int64_t* count_out,
               T* mean_out, T* var_out, float* cm_out, float* cr_out, int B, int L, int H, int G, int K,
               const Cfg& cfg, float b1, float b2, float om1, float om2, float eps, float dlog1, float dlog2,
               bool save_stats, Item* agg, cudaStream_t stream) {
  const long long chains = (long long)B * K;
  const int warps = cfg.warps;
  const dim3 grid((unsigned)((chains + warps - 1) / warps), (unsigned)cfg.ntiles, 1);
  const dim3 block(warps * kWarp, 1, 1);
  const int do_agg = (cfg.ntiles > 1);
  if (do_agg) {
    TdnFwdAggKernel<T, NV, 8><<<grid, block, 0, stream>>>(x, bos, pad, agg, L, H, G, K, cfg.tt, cfg.ntiles,
                                                          (int)chains, b1, b2);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }
  TdnFwdNormKernel<T, NV, 8><<<grid, block, 0, stream>>>(
      x, bos, pad, prev_count, prev_mean, prev_var, gamma, beta, y, count_out, mean_out, var_out,
      save_stats ? cm_out : nullptr, save_stats ? cr_out : nullptr, agg, L, H, G, K, cfg.tt, cfg.ntiles,
      (int)chains, om1, om2, eps, dlog1, dlog2);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template <typename T>
void LaunchT(const torch::Tensor& x, const c10::optional<torch::Tensor>& bos_mask, const torch::Tensor& prev_count,
             const torch::Tensor& prev_mean, const torch::Tensor& prev_var, const torch::Tensor& gamma,
             const torch::Tensor& beta, const c10::optional<torch::Tensor>& padding_mask, int64_t num_groups,
             double beta1, double beta2, double eps, torch::Tensor y_out, torch::Tensor count_out,
             torch::Tensor mean_out, torch::Tensor var_out, torch::Tensor cummean_out, torch::Tensor cumrstd_out,
             bool save_stats) {
  const int B = (int)x.size(0), L = (int)x.size(1), H = (int)x.size(2);
  TORCH_CHECK(num_groups > 0 && H % num_groups == 0 && H / num_groups <= kWarp * kMaxNv,
              "tdn fwd: num_groups must divide H, and a group is at most 256 channels");
  const int K = (int)num_groups, G = H / K;
  const Cfg cfg = ChooseCfg((long long)B * K, L);
  // The tile aggregates, one Item per (chain, tile), only when there is more than one tile a chain. Allocated per call:
  // the two kernels of a call are then the only ones that touch it, whatever else runs on the GPU at the same time.
  at::Tensor agg;
  if (cfg.ntiles > 1) {
    agg = at::empty({(long long)B * K * cfg.ntiles * 8}, x.options().dtype(at::kFloat));
  }
  Item* aggp = cfg.ntiles > 1 ? (Item*)agg.data_ptr() : nullptr;
  const unsigned char* bosp = bos_mask && bos_mask->defined() ? (const unsigned char*)bos_mask->data_ptr() : nullptr;
  const unsigned char* padp =
      padding_mask && padding_mask->defined() ? (const unsigned char*)padding_mask->data_ptr() : nullptr;
  const int64_t* pc = prev_count.defined() ? (const int64_t*)prev_count.data_ptr() : nullptr;
  const T* pm = prev_mean.defined() ? (const T*)prev_mean.data_ptr() : nullptr;
  const T* pv = prev_var.defined() ? (const T*)prev_var.data_ptr() : nullptr;
  // The recurrence's increment, and the bias correction is derived from it, so there is one constant, not two.
  // `1.f - (float)beta` subtracts two numbers of size 1, so it carries the float beta's own rounding divided by the
  // difference (1.3e-05 relative for beta1, 1.7e-04 for beta2). From the host's double, `(float)(1.0 - beta)` is the
  // correctly rounded constant, and it costs nothing: a kernel argument, not an instruction.
  const float om1 = (float)(1.0 - beta1);
  const float om2 = (float)(1.0 - beta2);
  // The bias correction `1 - beta**count` is `-expm1(count * log(beta))`, where `beta` is the decay the recurrence
  // actually has: in delta form that decay is `1 - om` in exact arithmetic, not `(float)beta`. So the log is taken
  // from `om`: `log1pf(-om)` is off the true log by the same relative 4.75e-08 as `om` itself. Taking it from
  // `(float)beta` instead would put the increment and the correction 1.3e-5 apart.
  const float dlog1 = log1pf(-om1);
  const float dlog2 = log1pf(-om2);
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  const int nv = (G + kWarp - 1) / kWarp;
  auto go = [&](auto tag) {
    constexpr int NV = decltype(tag)::value;
    LaunchOne<T, NV>((const T*)x.data_ptr(), bosp, padp, pc, pm, pv, (const T*)gamma.data_ptr(),
                     (const T*)beta.data_ptr(), (T*)y_out.data_ptr(),
                     count_out.defined() ? (int64_t*)count_out.data_ptr() : nullptr,
                     mean_out.defined() ? (T*)mean_out.data_ptr() : nullptr,
                     var_out.defined() ? (T*)var_out.data_ptr() : nullptr,
                     save_stats ? (float*)cummean_out.data_ptr() : nullptr,
                     save_stats ? (float*)cumrstd_out.data_ptr() : nullptr, B, L, H, G, K, cfg, (float)beta1,
                     (float)beta2, om1, om2, (float)eps, dlog1, dlog2, save_stats, aggp, stream);
  };
  switch (nv) {
    case 1:
      go(std::integral_constant<int, 1>{});
      break;
    case 2:
      go(std::integral_constant<int, 2>{});
      break;
    case 4:
      go(std::integral_constant<int, 4>{});
      break;
    case 8:
      go(std::integral_constant<int, 8>{});
      break;
    default:
      go(std::integral_constant<int, 0>{});
      break;
  }
}

}  // namespace

void TdnFwdGeneral(const at::Tensor& x, const c10::optional<at::Tensor>& bos, const at::Tensor& prev_count,
                   const at::Tensor& prev_mean, const at::Tensor& prev_var, const at::Tensor& gamma,
                   const at::Tensor& beta, const c10::optional<at::Tensor>& pad, int64_t num_groups, double beta1,
                   double beta2, double eps, bool save_stats, at::Tensor y, at::Tensor count, at::Tensor mean,
                   at::Tensor var, at::Tensor cummean_out, at::Tensor cumrstd_out) {
  TORCH_CHECK(x.is_cuda() && x.dim() == 3, "x must be [B, L, H] on the GPU");
  TORCH_CHECK(x.is_contiguous(), "x must be contiguous");
  const c10::cuda::CUDAGuard guard(x.device());
  // This path takes every shape the persistent forward declines: fp32, groups other than 64 channels, and the calls
  // it refuses at run time (a CUDA graph capture, more than 8 live streams, a failed record allocation).
  switch (x.scalar_type()) {
    case at::kBFloat16:
      LaunchT<__nv_bfloat16>(x, bos, prev_count, prev_mean, prev_var, gamma, beta, pad, num_groups, beta1, beta2, eps,
                             y, count, mean, var, cummean_out, cumrstd_out, save_stats);
      break;
    case at::kHalf:
      LaunchT<__half>(x, bos, prev_count, prev_mean, prev_var, gamma, beta, pad, num_groups, beta1, beta2, eps, y,
                      count, mean, var, cummean_out, cumrstd_out, save_stats);
      break;
    case at::kFloat:
      LaunchT<float>(x, bos, prev_count, prev_mean, prev_var, gamma, beta, pad, num_groups, beta1, beta2, eps, y,
                     count, mean, var, cummean_out, cumrstd_out, save_stats);
      break;
    default:
      TORCH_CHECK(false, "tdn fwd: unsupported dtype ", x.scalar_type());
  }
}

}  // namespace ops
}  // namespace xllm
