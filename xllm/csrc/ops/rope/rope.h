// The interface between run() (binding.cpp) and the kernels (rope.cu): everything is already reduced to raw
// pointers, element strides and a dtype code, so the host path does no ATen work and launches one kernel.
#pragma once
#include <cuda_runtime.h>

// One tensor of the call: [B, L, heads, R] with the last stride 1, in and out, strides in elements.
struct RopeTens {
  const void* in;
  void* out;
  int heads;
  long sB, sL, sH;  // input strides
  long oB, oL, oH;  // output strides
};

// dtype: 0 bf16, 1 fp16, 2 fp32. freqs: the complex64 table, its slice, fsL its stride in complex and fsP the stride
// between the pairs of one position (1 for xllm's own table, which is contiguous; any [L, R / 2] table is accepted).
void rope_launch(bool conj, int B, int L, int R, const void* freqs, long fsL, long fsP, int dtype, RopeTens t1,
                 RopeTens t2, cudaStream_t stream);
