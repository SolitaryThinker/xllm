// See odd.h. The class: cpg % 8 == 0 and vpg = cpg / 8 in [1, 32] and vpg NOT a power of two.
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <cstdint>
#include <type_traits>

#include "odd.h"

namespace xllm {
namespace ops {
namespace {

// ATen's tensors give at::BFloat16 / at::Half, not the CUDA built-in types: a comparison against
// __nv_bfloat16 alone silently takes the fp16 path and reinterprets bf16 bits as half.
template <typename T>
struct gn_is_bf16 : std::integral_constant<bool, std::is_same<T, __nv_bfloat16>::value ||
                                                     std::is_same<T, at::BFloat16>::value> {};

// ---------------------------------------------------------------- 8 channels a lane

__device__ __forceinline__ void unpack2(uint32_t w, float& a, float& b) {  // 2 x bf16
  a = __bfloat162float(__ushort_as_bfloat16((unsigned short)(w & 0xffffu)));
  b = __bfloat162float(__ushort_as_bfloat16((unsigned short)(w >> 16)));
}
__device__ __forceinline__ void unpack2h(uint32_t w, float& a, float& b) {  // 2 x half
  a = __half2float(__ushort_as_half((unsigned short)(w & 0xffffu)));
  b = __half2float(__ushort_as_half((unsigned short)(w >> 16)));
}
__device__ __forceinline__ uint32_t pack2(float a, float b) {  // 2 x bf16, round to nearest
  return (uint32_t)__bfloat16_as_ushort(__float2bfloat16(a)) |
         ((uint32_t)__bfloat16_as_ushort(__float2bfloat16(b)) << 16);
}
__device__ __forceinline__ uint32_t pack2h(float a, float b) {
  return (uint32_t)__half_as_ushort(__float2half_rn(a)) | ((uint32_t)__half_as_ushort(__float2half_rn(b)) << 16);
}

// 8 channels from 16 B (8 x 16-bit) or 32 B (8 x fp32) into 8 floats.
template <typename T>
__device__ __forceinline__ void ld8(const T* __restrict__ p, float* __restrict__ f) {
  if constexpr (std::is_same<T, float>::value) {
    const uint4 a = *reinterpret_cast<const uint4*>(p);
    const uint4 b = *reinterpret_cast<const uint4*>(p + 4);
    f[0] = __uint_as_float(a.x);
    f[1] = __uint_as_float(a.y);
    f[2] = __uint_as_float(a.z);
    f[3] = __uint_as_float(a.w);
    f[4] = __uint_as_float(b.x);
    f[5] = __uint_as_float(b.y);
    f[6] = __uint_as_float(b.z);
    f[7] = __uint_as_float(b.w);
  } else {
    const uint4 u = *reinterpret_cast<const uint4*>(p);
    if constexpr (gn_is_bf16<T>::value) {
      unpack2(u.x, f[0], f[1]);
      unpack2(u.y, f[2], f[3]);
      unpack2(u.z, f[4], f[5]);
      unpack2(u.w, f[6], f[7]);
    } else {
      unpack2h(u.x, f[0], f[1]);
      unpack2h(u.y, f[2], f[3]);
      unpack2h(u.z, f[4], f[5]);
      unpack2h(u.w, f[6], f[7]);
    }
  }
}

// Streaming store of 8 channels (the value is written once and not read again).
template <typename T>
__device__ __forceinline__ void st8(T* __restrict__ p, const float* __restrict__ f) {
  if constexpr (std::is_same<T, float>::value) {
    __stcs(reinterpret_cast<uint4*>(p),
           make_uint4(__float_as_uint(f[0]), __float_as_uint(f[1]), __float_as_uint(f[2]), __float_as_uint(f[3])));
    __stcs(reinterpret_cast<uint4*>(p + 4),
           make_uint4(__float_as_uint(f[4]), __float_as_uint(f[5]), __float_as_uint(f[6]), __float_as_uint(f[7])));
  } else {
    uint4 u;
    if constexpr (gn_is_bf16<T>::value) {
      u = make_uint4(pack2(f[0], f[1]), pack2(f[2], f[3]), pack2(f[4], f[5]), pack2(f[6], f[7]));
    } else {
      u = make_uint4(pack2h(f[0], f[1]), pack2h(f[2], f[3]), pack2h(f[4], f[5]), pack2h(f[6], f[7]));
    }
    __stcs(reinterpret_cast<uint4*>(p), u);
  }
}

// Sum of v over the group's vpg lanes, the group starting at lane `base_lane`, for a vpg that is never a power of two
// (24, 40, 56, 80, ... channels a group): every lane of the group is folded in, its own included. Every lane of the
// warp takes part; `v` itself is left alone, since the lanes read each other's value as the loop runs.
__device__ __forceinline__ float group_sum(float v, int vpg, int base_lane) {
  float t = 0.f;
  for (int i = 0; i < vpg; ++i) t += __shfl_sync(0xffffffffu, v, base_lane + i);
  return t;
}

// ---------------------------------------------------------------- forward

template <typename T, bool AFFINE, bool LAYER>
__global__ __launch_bounds__(256) void odd_fwd_kernel(const T* __restrict__ x, const T* __restrict__ gamma,
                                                        const T* __restrict__ beta, T* __restrict__ y,
                                                        float* __restrict__ mean_out, float* __restrict__ rstd_out,
                                                        int rows, int num_groups, int cpg, int vpg, float eps) {
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  const int gpw = 32 / vpg;           // groups per warp
  const int gidx = lane / vpg;        // which group inside the warp
  const int vidx = lane - gidx * vpg; // which vector of that group
  const int gpc = gpw * (blockDim.x >> 5);
  const int g = blockIdx.y * gpc + warp * gpw + gidx;
  const bool on = (gidx < gpw) && (g < num_groups);
  const long long C = (long long)num_groups * cpg;
  const long long ch = (long long)g * cpg + (long long)vidx * 8;
  const int base_lane = gidx * vpg;
  const float inv_n = 1.0f / (float)cpg;

  float ga[8], be[8];
#pragma unroll
  for (int i = 0; i < 8; ++i) {  // a lane outside every group gets 1 and 0, so its products stay 0
    ga[i] = 1.f;
    be[i] = 0.f;
  }
  if (AFFINE && on) {
    ld8(gamma + ch, ga);
    if (LAYER) ld8(beta + ch, be);
  }

  // One row: reduce this group over its vpg lanes from the 8 values in f, write the statistics and the output.
  auto do_row = [&](const float* __restrict__ f, long long rr) {
    float mean = 0.f, rstd;
    if (LAYER) {
      float ss = 0.f, sm = 0.f;
#pragma unroll
      for (int i = 0; i < 8; ++i) {
        ss += f[i] * f[i];
        sm += f[i];
      }
      mean = group_sum(sm, vpg, base_lane) * inv_n;
      float vs = 0.f;
#pragma unroll
      for (int i = 0; i < 8; ++i) {
        const float d = f[i] - mean;
        vs += d * d;
      }
      rstd = rsqrtf(group_sum(vs, vpg, base_lane) * inv_n + eps);
    } else {
      float ss = 0.f;
#pragma unroll
      for (int i = 0; i < 8; ++i) ss += f[i] * f[i];
      rstd = rsqrtf(group_sum(ss, vpg, base_lane) * inv_n + eps);
    }
    if (on && vidx == 0) {
      rstd_out[rr * num_groups + g] = rstd;
      if (LAYER) mean_out[rr * num_groups + g] = mean;
    }
    float o[8];
    if (AFFINE) {
      if (LAYER) {
#pragma unroll
        for (int i = 0; i < 8; ++i) o[i] = (f[i] - mean) * rstd * ga[i] + be[i];
      } else {
#pragma unroll
        for (int i = 0; i < 8; ++i) o[i] = f[i] * rstd * ga[i];
      }
    } else {
      if (LAYER) {
#pragma unroll
        for (int i = 0; i < 8; ++i) o[i] = (f[i] - mean) * rstd;
      } else {
#pragma unroll
        for (int i = 0; i < 8; ++i) o[i] = f[i] * rstd;
      }
    }
    if (on) st8(y + rr * C + ch, o);
  };

  // Two rows an iteration: both loads are issued before either is reduced, so a warp keeps two 16-byte accesses a
  // lane in flight. Four rows are slower (the registers cost more occupancy than the extra depth buys).
  const long long stride = gridDim.x;
  for (long long r = blockIdx.x; r < rows; r += 2 * stride) {
    const bool h1 = (r + stride) < rows;
    float a0[8], a1[8];
    if (on) {
      ld8(x + r * C + ch, a0);
      if (h1) ld8(x + (r + stride) * C + ch, a1);
    } else {
#pragma unroll
      for (int i = 0; i < 8; ++i) {
        a0[i] = 0.f;
        a1[i] = 0.f;
      }
    }
    do_row(a0, r);
    if (h1) do_row(a1, r + stride);
  }
}

// ---------------------------------------------------------------- backward

// dy and x_or_y in, dx out; with an affine transform also dgamma (and dbeta for the layer norm), each lane's 8
// channels accumulated over the rows it walks, summed across CTAs by atomics into an fp32 workspace.
template <typename T, bool AFFINE, bool LAYER, bool ME>
__global__ __launch_bounds__(256) void odd_bwd_kernel(
    const T* __restrict__ dy, const T* __restrict__ x_or_y, const float* __restrict__ mean_in,
    const float* __restrict__ rstd_in, const T* __restrict__ gamma, const T* __restrict__ beta, T* __restrict__ dx,
    float* __restrict__ dgamma_ws, float* __restrict__ dbeta_ws, int rows, int num_groups, int cpg, int vpg) {
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  const int gpw = 32 / vpg;
  const int gidx = lane / vpg;
  const int vidx = lane - gidx * vpg;
  const int gpc = gpw * (blockDim.x >> 5);
  const int g = blockIdx.y * gpc + warp * gpw + gidx;
  const bool on = (gidx < gpw) && (g < num_groups);
  const long long C = (long long)num_groups * cpg;
  const long long ch = (long long)g * cpg + (long long)vidx * 8;
  const int base_lane = gidx * vpg;
  const float inv_n = 1.0f / (float)cpg;

  float ga[8], be[8];
#pragma unroll
  for (int i = 0; i < 8; ++i) {  // a lane outside every group gets 1 and 0, so its products stay 0
    ga[i] = 1.f;
    be[i] = 0.f;
  }
  if (AFFINE && on) {
    ld8(gamma + ch, ga);
    if (LAYER) ld8(beta + ch, be);
  }
  float ag[8], ab[8];
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    ag[i] = 0.f;
    ab[i] = 0.f;
  }

  auto do_row = [&](const float* __restrict__ d, const float* __restrict__ v, long long rr) {
    // A lane past the last group has no statistics of its own: reading them would run past the end of the
    // [rows, num_groups] buffer on the last row. It contributes to no group's sum, so zero is safe here.
    const float rs = on ? rstd_in[rr * num_groups + g] : 0.f;
    const float mu = (LAYER && on) ? mean_in[rr * num_groups + g] : 0.f;
    float xh[8], dh[8];
    if (ME) {
      // x_hat recovered from y: (y - beta) / gamma, or y without an affine transform.
      if (AFFINE) {
        if (LAYER) {
#pragma unroll
          for (int i = 0; i < 8; ++i) xh[i] = (v[i] - be[i]) / ga[i];
        } else {
#pragma unroll
          for (int i = 0; i < 8; ++i) xh[i] = v[i] / ga[i];
        }
#pragma unroll
        for (int i = 0; i < 8; ++i) dh[i] = d[i] * ga[i];
      } else {
#pragma unroll
        for (int i = 0; i < 8; ++i) {
          xh[i] = v[i];
          dh[i] = d[i];
        }
      }
    } else {
      if (LAYER) {
#pragma unroll
        for (int i = 0; i < 8; ++i) xh[i] = (v[i] - mu) * rs;
      } else {
#pragma unroll
        for (int i = 0; i < 8; ++i) xh[i] = v[i] * rs;
      }
      if (AFFINE) {
#pragma unroll
        for (int i = 0; i < 8; ++i) dh[i] = d[i] * ga[i];
      } else {
#pragma unroll
        for (int i = 0; i < 8; ++i) dh[i] = d[i];
      }
    }

    float p = 0.f, sd = 0.f;
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      p += dh[i] * xh[i];
      if (LAYER) sd += dh[i];
    }
    const float proj = group_sum(p, vpg, base_lane) * inv_n;
    const float md = LAYER ? group_sum(sd, vpg, base_lane) * inv_n : 0.f;

    float o[8];
    if (LAYER) {
#pragma unroll
      for (int i = 0; i < 8; ++i) o[i] = (dh[i] - xh[i] * proj - md) * rs;
    } else {
#pragma unroll
      for (int i = 0; i < 8; ++i) o[i] = (dh[i] - xh[i] * proj) * rs;
    }
    if (on) st8(dx + rr * C + ch, o);

    if (AFFINE) {
#pragma unroll
      for (int i = 0; i < 8; ++i) {
        ag[i] += d[i] * xh[i];
        if (LAYER) ab[i] += d[i];
      }
    }
  };

  // One row an iteration: two tensors in, so the two loads of a row already keep a warp busy, and a second row in
  // flight costs registers the dgamma/dbeta accumulators need.
  for (long long r = blockIdx.x; r < rows; r += gridDim.x) {
    float d0[8], v0[8];
    if (on) {
      ld8(dy + r * C + ch, d0);
      ld8(x_or_y + r * C + ch, v0);
    } else {
#pragma unroll
      for (int i = 0; i < 8; ++i) {
        d0[i] = 0.f;
        v0[i] = 0.f;
      }
    }
    do_row(d0, v0, r);
  }

  if (AFFINE) {
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      if (on) {
        atomicAdd(dgamma_ws + ch + i, ag[i]);
        if (LAYER) atomicAdd(dbeta_ws + ch + i, ab[i]);
      }
    }
  }
}

template <typename T>
__global__ void odd_cast_kernel(const float* __restrict__ src, T* __restrict__ dst, int n) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) dst[i] = (T)src[i];
}

// ---------------------------------------------------------------- host

struct Plan {
  int vpg, gpw, warps, threads, chunks, rb;
};

Plan odd_plan(long long num_groups, long long cpg, long long rows) {
  Plan p;
  p.vpg = (int)(cpg / 8);
  p.gpw = 32 / p.vpg;
  long long warps = (num_groups + p.gpw - 1) / p.gpw;
  if (warps > 8) warps = 8;
  if (warps < 1) warps = 1;
  p.warps = (int)warps;
  p.threads = 32 * p.warps;
  const long long gpc = (long long)p.gpw * p.warps;
  p.chunks = (int)((num_groups + gpc - 1) / gpc);
  // About eight CTAs per SM, each walking a grid-stride slice of the rows.
  long long target = (long long)at::cuda::getCurrentDeviceProperties()->multiProcessorCount * (2048 / p.threads);
  long long rb = (target + p.chunks - 1) / p.chunks;
  if (rb > rows) rb = rows;
  if (rb < 1) rb = 1;
  p.rb = (int)rb;
  return p;
}

bool odd_class(long long cpg) {
  if (cpg <= 0 || cpg % 8 != 0) return false;
  const long long vpg = cpg / 8;
  if (vpg < 1 || vpg > 32) return false;
  return (vpg & (vpg - 1)) != 0;
}

template <typename T>
void fwd_impl(bool layer, bool affine, const at::Tensor& x, const at::Tensor& gamma, const at::Tensor& beta,
              at::Tensor y, at::Tensor mean_out, at::Tensor rstd_out, long long rows, long long num_groups,
              long long cpg, float eps, cudaStream_t stream) {
  const Plan p = odd_plan(num_groups, cpg, rows);
  const T* xd = x.data_ptr<T>();
  T* yd = y.data_ptr<T>();
  float* md = layer ? mean_out.data_ptr<float>() : nullptr;
  float* rd = rstd_out.data_ptr<float>();
  const T* gd = affine ? gamma.data_ptr<T>() : nullptr;
  const T* bd = (affine && layer) ? beta.data_ptr<T>() : nullptr;
  const dim3 grid(p.rb, p.chunks), block(p.threads);
  if (affine) {
    if (layer)
      odd_fwd_kernel<T, true, true><<<grid, block, 0, stream>>>(xd, gd, bd, yd, md, rd, (int)rows, (int)num_groups,
                                                                  (int)cpg, p.vpg, eps);
    else
      odd_fwd_kernel<T, true, false><<<grid, block, 0, stream>>>(xd, gd, bd, yd, md, rd, (int)rows, (int)num_groups,
                                                                   (int)cpg, p.vpg, eps);
  } else {
    if (layer)
      odd_fwd_kernel<T, false, true><<<grid, block, 0, stream>>>(xd, gd, bd, yd, md, rd, (int)rows, (int)num_groups,
                                                                   (int)cpg, p.vpg, eps);
    else
      odd_fwd_kernel<T, false, false><<<grid, block, 0, stream>>>(xd, gd, bd, yd, md, rd, (int)rows,
                                                                    (int)num_groups, (int)cpg, p.vpg, eps);
  }
}

template <typename T>
void bwd_impl(bool layer, bool affine, bool me, const at::Tensor& dy, const at::Tensor& x_or_y,
              const at::Tensor& mean, const at::Tensor& rstd, const at::Tensor& gamma, const at::Tensor& beta,
              at::Tensor dx, at::Tensor dgamma, at::Tensor dbeta, long long rows, long long num_groups, long long cpg,
              cudaStream_t stream) {
  const Plan p = odd_plan(num_groups, cpg, rows);
  const T* dyd = dy.data_ptr<T>();
  const T* xd = x_or_y.data_ptr<T>();
  const float* md = layer ? mean.data_ptr<float>() : nullptr;
  const float* rd = rstd.data_ptr<float>();
  T* dxd = dx.data_ptr<T>();
  const T* gd = affine ? gamma.data_ptr<T>() : nullptr;
  const T* bd = (affine && layer) ? beta.data_ptr<T>() : nullptr;
  float* dws = nullptr;
  float* bws = nullptr;
  at::Tensor dws_t, bws_t;
  if (affine) {
    dws_t = at::zeros({cpg * num_groups}, x_or_y.options().dtype(at::kFloat));
    dws = dws_t.data_ptr<float>();
    if (layer) {
      bws_t = at::zeros({cpg * num_groups}, x_or_y.options().dtype(at::kFloat));
      bws = bws_t.data_ptr<float>();
    }
  }
  const dim3 grid(p.rb, p.chunks), block(p.threads);
#define GN_ODD_BWD(A, L, M)                                                                                     \
  odd_bwd_kernel<T, A, L, M><<<grid, block, 0, stream>>>(dyd, xd, md, rd, gd, bd, dxd, dws, bws, (int)rows,  \
                                                            (int)num_groups, (int)cpg, p.vpg)
  if (affine) {
    if (layer) {
      if (me) GN_ODD_BWD(true, true, true);
      else GN_ODD_BWD(true, true, false);
    } else {
      if (me) GN_ODD_BWD(true, false, true);
      else GN_ODD_BWD(true, false, false);
    }
  } else {
    if (layer) {
      if (me) GN_ODD_BWD(false, true, true);
      else GN_ODD_BWD(false, true, false);
    } else {
      if (me) GN_ODD_BWD(false, false, true);
      else GN_ODD_BWD(false, false, false);
    }
  }
#undef GN_ODD_BWD
  if (affine) {
    const int n = (int)(cpg * num_groups);
    const int nb = (n + 255) / 256;
    odd_cast_kernel<T><<<nb, 256, 0, stream>>>(dws, dgamma.data_ptr<T>(), n);
    if (layer) odd_cast_kernel<T><<<nb, 256, 0, stream>>>(bws, dbeta.data_ptr<T>(), n);
  }
}

}  // namespace

bool GroupNormOddRun(int64_t mode, int64_t norm, int64_t num_groups, double eps, bool memory_efficient,
                     const at::Tensor& x_or_y, const at::Tensor& dy, const at::Tensor& mean, const at::Tensor& rstd,
                     const at::Tensor& gamma, const at::Tensor& beta, const at::Tensor& y, const at::Tensor& mean_out,
                     const at::Tensor& rstd_out, const at::Tensor& dx, const at::Tensor& dgamma,
                     const at::Tensor& dbeta) {
  if (x_or_y.dim() != 3 || num_groups < 1) return false;
  const long long channels = x_or_y.size(-1);
  if (channels % num_groups != 0) return false;
  const long long cpg = channels / num_groups;
  if (!odd_class(cpg)) return false;
  const long long rows = x_or_y.numel() / channels;
  const bool layer = norm != 0;
  const bool affine = gamma.numel() > 0;

  const at::cuda::OptionalCUDAGuard guard(at::device_of(x_or_y));
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  // float, half and bf16 only: the 16-byte path reinterprets bytes, so any other dtype is declined.
  const auto dt = x_or_y.scalar_type();
#define GN_ODD_TYPES(FN, ...)                                   \
  if (dt == at::kFloat) FN<float>(__VA_ARGS__);                 \
  else if (dt == at::kHalf) FN<at::Half>(__VA_ARGS__);          \
  else if (dt == at::kBFloat16) FN<at::BFloat16>(__VA_ARGS__);  \
  else return false;
  if (mode == 0) {
    GN_ODD_TYPES(fwd_impl, layer, affine, x_or_y, gamma, beta, y, mean_out, rstd_out, rows, num_groups, cpg,
                 (float)eps, stream)
  } else {
    GN_ODD_TYPES(bwd_impl, layer, affine, memory_efficient, dy, x_or_y, mean, rstd, gamma, beta, dx, dgamma, dbeta,
                 rows, num_groups, cpg, stream)
  }
#undef GN_ODD_TYPES
  return true;
}

}  // namespace ops
}  // namespace xllm
