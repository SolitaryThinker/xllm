// Entry point of the TMA group-norm backward (bwd_tma.cu). It returns false, having done nothing (no launch, no
// output written), when the call is outside what the kernel handles or the device refuses the launch; the caller then
// takes another path.
#pragma once

#include <torch/extension.h>

namespace gn_tma {

// Backward pass: dx and, with gamma, dgamma (and dbeta for the layer norm), in one pass over x_or_y and dy.
// With gamma it takes a [grid, C] (layer norm: [grid, 2C]) fp32 workspace for the CTA partials from the caching
// allocator, on the current stream.
bool backward(bool layer, int64_t num_groups, bool memory_efficient, const at::Tensor& x_or_y, const at::Tensor& dy,
              const at::Tensor& mean, const at::Tensor& rstd, const at::Tensor& gamma, const at::Tensor& beta,
              at::Tensor& dx, at::Tensor& dgamma, at::Tensor& dbeta);

}  // namespace gn_tma
