// Declarations shared by awm_softmax_binding.cpp (dispatch) and awm_chunk_softmax.cu (kernels).
#pragma once
#include <cstddef>
#include <cuda_runtime.h>

namespace awm {

constexpr int PAD = 8;                    // padding of a shared tile row (elements)
constexpr size_t kMaxShared = 200u * 1024u;

struct FwdArgs {
  const void* key;
  const void* mask;
  void* kout;
  float* lse;
  long long inBStr, inTStr, inKStr;
  long long outBStr, outKStr, outL;
  long long maskBStr;
  long long lseBStr, lseKStr, lseNStr;
  int B, K, nbChan, C, cMul, lseMaxC, slotBase, nTasks;
};

struct BwdArgs {
  const void* key;
  const void* mask;
  const float* lse;
  const void* dkkey;
  const float* dlse;
  void* dkey;
  long long inBStr, inTStr, inKStr;
  long long maskBStr;
  long long lseBStr, lseKStr, lseNStr;
  long long dkCL;  // elements per channel row of dkkey (L - C)
  long long dkKStr;
  int B, K, nbChan, C, N, nTasks;
};

// dtype: 0 bf16, 1 fp16, 2 fp32. ST: channels per CTA; P: CTAs of a cluster, which hold one slice each of the
// chunk's time (1: no cluster). Returns 0 when it launched.
int fwd_launch(const FwdArgs& a0, const FwdArgs& a1, int has1, int dtype, int ST, int P, cudaStream_t st);
int bwd_launch(const BwdArgs& a, int dtype, int ST, cudaStream_t st);

}  // namespace awm
