// Launch interface of the flat forward kernels (fwd_flat.cu): group RMS / layer norm, any group count.
#pragma once

namespace gn {

struct FwdParams {
  const void* x;      // [rows, groups * cpg], contiguous
  const void* gamma;  // [groups * cpg] or null (no affine transform)
  const void* beta;   // [groups * cpg], null unless the layer norm with an affine transform
  void* y;            // [rows, groups * cpg]
  void* mean;         // [rows, groups] fp32, null unless the layer norm
  void* rstd;         // [rows, groups] fp32
  unsigned long long rows;
  unsigned groups;    // groups per row
  unsigned cpg;       // channels per group (>= 1)
  float eps;
  int dtype;          // 0 = bfloat16, 1 = float16, 2 = float32
  int layer;          // 1 = layer norm (subtract the mean, write mean), 0 = RMS norm. Not `beta != null`:
                      // a layer norm without an affine transform has no beta.
  int grid_bpsm;      // cap on the grid, in CTAs of kBlock per SM
};

// One kernel launch. `stream` is a cudaStream_t.
void fwd_launch(const FwdParams& p, void* stream);

// The same kernel with half as many lanes per group (more whole groups per warp, Q chunks in flight).
void fwd_launch_half(const FwdParams& p, void* stream);

}  // namespace gn
