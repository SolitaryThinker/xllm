// The conversion, vector and in-warp reduction helpers that fwd_by_width.cu and bwd_by_width.cu share. Header only;
// every function is __device__ __forceinline__, and each kernel keeps its own block-wide reduction (which needs its
// shared-memory shape) beside it.
#pragma once

#include <cstdint>

#include <cuda_bf16.h>
#include <cuda_fp16.h>

namespace gn {
namespace width {

// ---------------------------------------------------------------- conversions and vectors

__device__ __forceinline__ float to_f(float v) { return v; }
__device__ __forceinline__ float to_f(__half v) { return __half2float(v); }
__device__ __forceinline__ float to_f(__nv_bfloat16 v) { return __bfloat162float(v); }

__device__ __forceinline__ void from_f(float& d, float s) { d = s; }
__device__ __forceinline__ void from_f(__half& d, float s) { d = __float2half(s); }
__device__ __forceinline__ void from_f(__nv_bfloat16& d, float s) { d = __float2bfloat16(s); }

template <int BYTES>
struct RawBytes;
template <>
struct RawBytes<16> {
  using type = uint4;
};
template <>
struct RawBytes<8> {
  using type = uint2;
};
template <>
struct RawBytes<4> {
  using type = unsigned int;
};
template <>
struct RawBytes<2> {
  using type = unsigned short;
};
template <>
struct RawBytes<1> {
  using type = unsigned char;
};

template <typename T, int V>
struct alignas(sizeof(T) * V) Vec {
  T v[V];
};

template <typename T, int V>
__device__ __forceinline__ Vec<T, V> load_stream(const T* p) {
  using R = typename RawBytes<sizeof(T) * V>::type;
  union {
    R r;
    Vec<T, V> v;
  } u;
  u.r = __ldcs(reinterpret_cast<const R*>(p));
  return u.v;
}

template <typename T, int V>
__device__ __forceinline__ void store_stream(T* p, const Vec<T, V>& val) {
  using R = typename RawBytes<sizeof(T) * V>::type;
  union {
    R r;
    Vec<T, V> v;
  } u;
  u.v = val;
  __stcs(reinterpret_cast<R*>(p), u.r);
}

// gamma and beta are read by every row: keep them cached.
template <typename T, int V>
__device__ __forceinline__ Vec<T, V> load_cached(const T* p) {
  using R = typename RawBytes<sizeof(T) * V>::type;
  union {
    R r;
    Vec<T, V> v;
  } u;
  u.r = __ldg(reinterpret_cast<const R*>(p));
  return u.v;
}

__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
  for (int off = 16; off > 0; off >>= 1) v += __shfl_xor_sync(0xffffffffu, v, off);
  return v;
}

// The sum over the L lanes of this lane's group: every lane of the group ends with the group's sum.
template <int L>
__device__ __forceinline__ float group_sum(float s, int lane) {
  if constexpr ((L & (L - 1)) == 0) {
#pragma unroll
    for (int off = L / 2; off > 0; off >>= 1) s += __shfl_xor_sync(0xffffffffu, s, off, L);
  } else {
    const int base = (lane / L) * L;
    float t = 0.f;
#pragma unroll
    for (int i = 0; i < L; ++i) t += __shfl_sync(0xffffffffu, s, base + i);
    s = t;
  }
  return s;
}

__device__ __forceinline__ float group_sum_rt(float s, int lane, int L) {
  const int base = (lane / L) * L;
  float t = 0.f;
  for (int i = 0; i < L; ++i) t += __shfl_sync(0xffffffffu, s, base + i);
  return t;
}

// The same, for two values at once (the backward's two group statistics): every lane gets both sums.
template <int L>
__device__ __forceinline__ void group_sum2(float& a, float& b, int lane, int l_rt) {
  const int l = (L > 0) ? L : l_rt;
  if constexpr (L > 0 && (L & (L - 1)) == 0) {
#pragma unroll
    for (int off = L / 2; off > 0; off >>= 1) {
      a += __shfl_xor_sync(0xffffffffu, a, off, L);
      b += __shfl_xor_sync(0xffffffffu, b, off, L);
    }
  } else {
    const int base = (lane / l) * l;
    float ta = 0.f, tb = 0.f;
    for (int i = 0; i < l; ++i) {
      ta += __shfl_sync(0xffffffffu, a, base + i);
      tb += __shfl_sync(0xffffffffu, b, base + i);
    }
    a = ta;
    b = tb;
  }
}

}  // namespace width
}  // namespace gn
