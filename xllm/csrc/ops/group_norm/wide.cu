// See wide.h. The class: cpg % 8 == 0 and vpg = cpg / 8 >= 64 (cpg >= 512).
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <cstdint>
#include <type_traits>

#include "wide.h"

namespace xllm {
namespace ops {
namespace {

// ATen's tensors give at::BFloat16 / at::Half, not the CUDA built-in types.
template <typename T>
struct gn_is_bf16 : std::integral_constant<bool, std::is_same<T, __nv_bfloat16>::value ||
                                                     std::is_same<T, at::BFloat16>::value> {};

__device__ __forceinline__ void unpack2(uint32_t w, float& a, float& b) {  // 2 x bf16
  a = __bfloat162float(__ushort_as_bfloat16((unsigned short)(w & 0xffffu)));
  b = __bfloat162float(__ushort_as_bfloat16((unsigned short)(w >> 16)));
}
__device__ __forceinline__ void unpack2h(uint32_t w, float& a, float& b) {  // 2 x half
  a = __half2float(__ushort_as_half((unsigned short)(w & 0xffffu)));
  b = __half2float(__ushort_as_half((unsigned short)(w >> 16)));
}
__device__ __forceinline__ uint32_t pack2(float a, float b) {
  return (uint32_t)__bfloat16_as_ushort(__float2bfloat16(a)) |
         ((uint32_t)__bfloat16_as_ushort(__float2bfloat16(b)) << 16);
}
__device__ __forceinline__ uint32_t pack2h(float a, float b) {
  return (uint32_t)__half_as_ushort(__float2half_rn(a)) | ((uint32_t)__half_as_ushort(__float2half_rn(b)) << 16);
}

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

__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
  for (int off = 16; off > 0; off >>= 1) v += __shfl_xor_sync(0xffffffffu, v, off);
  return v;
}

// Sum across the CTA, given back to every thread. sh needs nwarp floats (one) or 2 * nwarp (two), plus the totals.
// The leading barrier is required: the slot this call writes is the slot the previous call's broadcast was read from,
// and a warp that gets a whole row ahead would otherwise overwrite a partial another warp is still reading as the
// group total. It matters twice over because the layer-norm forward calls this twice back to back (mean, then
// variance) and the layer-norm backward calls block_sum2 then block_sum1 in the same row. The extra barrier a
// reduction costs almost nothing.
__device__ __forceinline__ float block_sum1(float v, float* sh, int nwarp) {
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  v = warp_sum(v);
  __syncthreads();
  if (lane == 0) sh[warp] = v;
  __syncthreads();
  if (warp == 0) {
    float t = (lane < nwarp) ? sh[lane] : 0.f;
    t = warp_sum(t);
    if (lane == 0) sh[0] = t;
  }
  __syncthreads();
  return sh[0];
}

__device__ __forceinline__ void block_sum2(float& a, float& b, float* sh, int nwarp) {
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  a = warp_sum(a);
  b = warp_sum(b);
  __syncthreads();
  if (lane == 0) {
    sh[warp] = a;
    sh[nwarp + warp] = b;
  }
  __syncthreads();
  if (warp == 0) {
    float ta = (lane < nwarp) ? sh[lane] : 0.f;
    float tb = (lane < nwarp) ? sh[nwarp + lane] : 0.f;
    ta = warp_sum(ta);
    tb = warp_sum(tb);
    if (lane == 0) {
      sh[0] = ta;
      sh[nwarp] = tb;
    }
  }
  __syncthreads();
  a = sh[0];
  b = sh[nwarp];
}

// ---------------------------------------------------------------- forward

template <typename T, bool AFFINE, bool LAYER>
__global__ __launch_bounds__(1024) void wide_fwd_kernel(const T* __restrict__ x, const T* __restrict__ gamma,
                                                          const T* __restrict__ beta, T* __restrict__ y,
                                                          float* __restrict__ mean_out, float* __restrict__ rstd_out,
                                                          long long rows, int num_groups, int cpg, float eps) {
  extern __shared__ float sh[];
  const int tid = threadIdx.x;
  const int nwarp = blockDim.x >> 5;
  const int g = blockIdx.y;
  const long long C = (long long)num_groups * cpg;
  const int vch = tid * 8;  // the channels of this thread's vector, inside the group
  const bool on = vch < cpg;
  const long long ch = (long long)g * cpg + vch;
  const float inv_n = 1.0f / (float)cpg;

  float ga[8], be[8];
#pragma unroll
  for (int i = 0; i < 8; ++i) {  // a thread past the group's vectors gets 1 and 0, so its products stay 0
    ga[i] = 1.f;
    be[i] = 0.f;
  }
  if (AFFINE && on) {
    ld8(gamma + ch, ga);
    if (LAYER) ld8(beta + ch, be);
  }

  for (long long r = blockIdx.x; r < rows; r += gridDim.x) {
    float f[8];
    if (on) ld8(x + r * C + ch, f);
    else {
#pragma unroll
      for (int i = 0; i < 8; ++i) f[i] = 0.f;
    }
    float mean = 0.f, ss = 0.f;
    if (LAYER) {
      float sm = 0.f;
#pragma unroll
      for (int i = 0; i < 8; ++i) sm += f[i];
      mean = block_sum1(sm, sh, nwarp) * inv_n;
      // The centred squares, not E[x^2] - mean^2; and a thread past the group's vectors centres nothing (mk = 0),
      // or its 8 * mean^2 would land in the group's variance.
      const float mk = on ? mean : 0.f;
      float vs = 0.f;
#pragma unroll
      for (int i = 0; i < 8; ++i) {
        const float d = f[i] - mk;
        vs += d * d;
      }
      ss = block_sum1(vs, sh, nwarp) * inv_n + eps;
    } else {
#pragma unroll
      for (int i = 0; i < 8; ++i) ss += f[i] * f[i];
      ss = block_sum1(ss, sh, nwarp) * inv_n + eps;
    }
    const float rstd = rsqrtf(ss);
    if (tid == 0) {
      rstd_out[r * num_groups + g] = rstd;
      if (LAYER) mean_out[r * num_groups + g] = mean;
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
    if (on) st8(y + r * C + ch, o);
  }
}

// ---------------------------------------------------------------- backward

template <typename T, bool AFFINE, bool LAYER, bool ME>
__global__ __launch_bounds__(1024) void wide_bwd_kernel(
    const T* __restrict__ dy, const T* __restrict__ x_or_y, const float* __restrict__ mean_in,
    const float* __restrict__ rstd_in, const T* __restrict__ gamma, const T* __restrict__ beta, T* __restrict__ dx,
    float* __restrict__ dgamma_ws, float* __restrict__ dbeta_ws, long long rows, int num_groups, int cpg) {
  extern __shared__ float sh[];
  const int tid = threadIdx.x;
  const int nwarp = blockDim.x >> 5;
  const int g = blockIdx.y;
  const long long C = (long long)num_groups * cpg;
  const int vch = tid * 8;
  const bool on = vch < cpg;
  const long long ch = (long long)g * cpg + vch;
  const float inv_n = 1.0f / (float)cpg;

  float ga[8], be[8];
#pragma unroll
  for (int i = 0; i < 8; ++i) {  // a thread past the group's vectors gets 1 and 0, so its products stay 0
    ga[i] = 1.f;
    be[i] = 0.f;
  }
  if (AFFINE && on) {
    ld8(gamma + ch, ga);
    if (LAYER) ld8(beta + ch, be);
  }
  // The memory_efficient recovery is (y - beta) / gamma. Dividing eight times a row would be eight divisions a lane a
  // row by a value that does not change, so the reciprocal of gamma is taken once a call and the row uses eight
  // multiplies. Dead code in every other instantiation.
  float ig[8];
  if constexpr (ME && AFFINE) {
#pragma unroll
    for (int i = 0; i < 8; ++i) ig[i] = 1.f / ga[i];
  }
  float ag[8], ab[8];
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    ag[i] = 0.f;
    ab[i] = 0.f;
  }

  for (long long r = blockIdx.x; r < rows; r += gridDim.x) {
    const long long off = r * C + ch;
    float d[8], v[8];
    if (on) {
      ld8(dy + off, d);
      ld8(x_or_y + off, v);
    } else {
#pragma unroll
      for (int i = 0; i < 8; ++i) {
        d[i] = 0.f;
        v[i] = 0.f;
      }
    }
    const float rs = on ? rstd_in[r * num_groups + g] : 0.f;
    const float mu = (LAYER && on) ? mean_in[r * num_groups + g] : 0.f;

    // x_hat and dy*gamma are recomputed after the reduction rather than kept live across it: only d and v have to
    // survive the CTA's two barriers, which is what keeps the register count low enough for another CTA a SM.
    auto xh_at = [&](int i) -> float {
      if (ME) return AFFINE ? (LAYER ? (v[i] - be[i]) * ig[i] : v[i] * ig[i]) : v[i];
      return LAYER ? (v[i] - mu) * rs : v[i] * rs;
    };
    auto dh_at = [&](int i) -> float { return AFFINE ? d[i] * ga[i] : d[i]; };

    float p = 0.f, sd = 0.f;
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      const float xh = xh_at(i);
      const float dh = dh_at(i);
      p += dh * xh;
      if (LAYER) sd += dh;
    }
    float proj, md = 0.f;
    if (LAYER) {
      block_sum2(p, sd, sh, nwarp);
      proj = p * inv_n;
      md = sd * inv_n;
    } else {
      proj = block_sum1(p, sh, nwarp) * inv_n;
    }

    float o[8];
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      const float xh = xh_at(i);
      const float dh = dh_at(i);
      o[i] = LAYER ? (dh - xh * proj - md) * rs : (dh - xh * proj) * rs;
      if (AFFINE) {
        ag[i] += d[i] * xh;
        if (LAYER) ab[i] += d[i];
      }
    }
    if (on) st8(dx + off, o);
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
__global__ void wide_cast_kernel(const float* __restrict__ src, T* __restrict__ dst, int n) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) dst[i] = (T)src[i];
}

// ---------------------------------------------------------------- host

struct WPlan {
  int warps, threads, chunks, rc;
};

WPlan wide_plan(long long num_groups, long long cpg, long long rows) {
  WPlan p;
  const long long vpg = cpg / 8;
  long long warps = (vpg + 31) / 32;
  if (warps > 32) warps = 32;
  if (warps < 1) warps = 1;
  p.warps = (int)warps;
  p.threads = 32 * p.warps;
  p.chunks = 1;  // a CTA covers a whole group
  long long target = (long long)at::cuda::getCurrentDeviceProperties()->multiProcessorCount *
                     (2048 / p.threads);  // CTAs to fill the SMs
  long long rc = (target + num_groups - 1) / num_groups;
  if (rc > rows) rc = rows;
  if (rc < 1) rc = 1;
  p.rc = (int)rc;
  return p;
}

bool wide_class(long long cpg) {
  if (cpg <= 0 || cpg % 8 != 0) return false;
  const long long vpg = cpg / 8;
  return vpg >= 64 && vpg <= 32 * 32;  // at most 32 warps hold a group
}

template <typename T>
void fwd_impl(bool layer, bool affine, const at::Tensor& x, const at::Tensor& gamma, const at::Tensor& beta,
              at::Tensor y, at::Tensor mean_out, at::Tensor rstd_out, long long rows, long long num_groups,
              long long cpg, float eps, cudaStream_t stream) {
  const WPlan p = wide_plan(num_groups, cpg, rows);
  const T* xd = x.data_ptr<T>();
  T* yd = y.data_ptr<T>();
  float* md = layer ? mean_out.data_ptr<float>() : nullptr;
  float* rd = rstd_out.data_ptr<float>();
  const T* gd = affine ? gamma.data_ptr<T>() : nullptr;
  const T* bd = (affine && layer) ? beta.data_ptr<T>() : nullptr;
  const dim3 grid(p.rc, (unsigned)num_groups), block(p.threads);
  const size_t shm = sizeof(float) * 2 * p.warps + 64;
  if (affine) {
    if (layer)
      wide_fwd_kernel<T, true, true><<<grid, block, shm, stream>>>(xd, gd, bd, yd, md, rd, rows, (int)num_groups,
                                                                     (int)cpg, eps);
    else
      wide_fwd_kernel<T, true, false><<<grid, block, shm, stream>>>(xd, gd, bd, yd, md, rd, rows, (int)num_groups,
                                                                      (int)cpg, eps);
  } else {
    if (layer)
      wide_fwd_kernel<T, false, true><<<grid, block, shm, stream>>>(xd, gd, bd, yd, md, rd, rows, (int)num_groups,
                                                                      (int)cpg, eps);
    else
      wide_fwd_kernel<T, false, false><<<grid, block, shm, stream>>>(xd, gd, bd, yd, md, rd, rows, (int)num_groups,
                                                                       (int)cpg, eps);
  }
}

template <typename T>
void bwd_impl(bool layer, bool affine, bool me, const at::Tensor& dy, const at::Tensor& x_or_y,
              const at::Tensor& mean, const at::Tensor& rstd, const at::Tensor& gamma, const at::Tensor& beta,
              at::Tensor dx, at::Tensor dgamma, at::Tensor dbeta, long long rows, long long num_groups, long long cpg,
              cudaStream_t stream) {
  const WPlan p = wide_plan(num_groups, cpg, rows);
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
  const dim3 grid(p.rc, (unsigned)num_groups), block(p.threads);
  const size_t shm = sizeof(float) * 2 * p.warps + 64;
#define GN_WIDE_BWD(A, L, M)                                                                              \
  wide_bwd_kernel<T, A, L, M><<<grid, block, shm, stream>>>(dyd, xd, md, rd, gd, bd, dxd, dws, bws, rows, \
                                                              (int)num_groups, (int)cpg)
  if (affine) {
    if (layer) {
      if (me) GN_WIDE_BWD(true, true, true);
      else GN_WIDE_BWD(true, true, false);
    } else {
      if (me) GN_WIDE_BWD(true, false, true);
      else GN_WIDE_BWD(true, false, false);
    }
  } else {
    if (layer) {
      if (me) GN_WIDE_BWD(false, true, true);
      else GN_WIDE_BWD(false, true, false);
    } else {
      if (me) GN_WIDE_BWD(false, false, true);
      else GN_WIDE_BWD(false, false, false);
    }
  }
#undef GN_WIDE_BWD
  if (affine) {
    const int n = (int)(cpg * num_groups);
    const int nb = (n + 255) / 256;
    wide_cast_kernel<T><<<nb, 256, 0, stream>>>(dws, dgamma.data_ptr<T>(), n);
    if (layer) wide_cast_kernel<T><<<nb, 256, 0, stream>>>(bws, dbeta.data_ptr<T>(), n);
  }
}

}  // namespace

bool GroupNormWideRun(int64_t mode, int64_t norm, int64_t num_groups, double eps, bool memory_efficient,
                      const at::Tensor& x_or_y, const at::Tensor& dy, const at::Tensor& mean, const at::Tensor& rstd,
                      const at::Tensor& gamma, const at::Tensor& beta, const at::Tensor& y, const at::Tensor& mean_out,
                      const at::Tensor& rstd_out, const at::Tensor& dx, const at::Tensor& dgamma,
                      const at::Tensor& dbeta) {
  if (x_or_y.dim() != 3 || num_groups < 1) return false;
  const long long channels = x_or_y.size(-1);
  if (channels % num_groups != 0) return false;
  const long long cpg = channels / num_groups;
  if (!wide_class(cpg)) return false;
  const long long rows = x_or_y.numel() / channels;
  const bool layer = norm != 0;
  const bool affine = gamma.numel() > 0;

  const at::cuda::OptionalCUDAGuard guard(at::device_of(x_or_y));
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  // float, half and bf16 only: the 16-byte path reinterprets bytes, so any other dtype is declined.
  const auto dt = x_or_y.scalar_type();
#define GN_WIDE_TYPES(FN, ...)                                  \
  if (dt == at::kFloat) FN<float>(__VA_ARGS__);                 \
  else if (dt == at::kHalf) FN<at::Half>(__VA_ARGS__);          \
  else if (dt == at::kBFloat16) FN<at::BFloat16>(__VA_ARGS__);  \
  else return false;
  if (mode == 0) {
    GN_WIDE_TYPES(fwd_impl, layer, affine, x_or_y, gamma, beta, y, mean_out, rstd_out, rows, num_groups, cpg,
                  (float)eps, stream)
  } else {
    GN_WIDE_TYPES(bwd_impl, layer, affine, memory_efficient, dy, x_or_y, mean, rstd, gamma, beta, dx, dgamma, dbeta,
                  rows, num_groups, cpg, stream)
  }
#undef GN_WIDE_TYPES
  return true;
}

}  // namespace ops
}  // namespace xllm
