// Registers the wide kernels (a group of 512 channels or more, wide.cu) and the odd kernels (a group of a
// non-power-of-two number of 16-byte vectors, odd.cu), for the forward pass and for the one-pass backward.
#include "dispatch.h"
#include "odd.h"   // xllm::ops::GroupNormOddRun
#include "wide.h"  // xllm::ops::GroupNormWideRun

namespace {

// Both entry points answer false for every shape whose group width is not their class, and check it before touching
// anything, so this node claims the whole call and declines in `run`. The wide class is tried first, then the odd one.
bool wide_odd_handles(const gn::Shape&) { return true; }

bool wide_odd_run(const gn::Shape& s, const gn::Call& c) {
  const int64_t mode = s.mode == gn::kFwd ? 0 : 1;
  if (xllm::ops::GroupNormWideRun(mode, s.norm, s.groups, c.eps, s.memory_efficient, *c.x_or_y, *c.dy, *c.mean,
                                 *c.rstd, *c.gamma, *c.beta, *c.y, *c.mean_out, *c.rstd_out, *c.dx, *c.dgamma,
                                 *c.dbeta))
    return true;
  return xllm::ops::GroupNormOddRun(mode, s.norm, s.groups, c.eps, s.memory_efficient, *c.x_or_y, *c.dy, *c.mean,
                                    *c.rstd, *c.gamma, *c.beta, *c.y, *c.mean_out, *c.rstd_out, *c.dx, *c.dgamma,
                                    *c.dbeta);
}

// The forward sits at priority 15, below the lane-group forward (20) and the by-width forward (24). The backward sits
// at 9, below the one-pass backward (12) and the dense-lane affine backward (10), which are faster on the affine
// shapes that reach this far down.
const bool wide_odd_fwd_registered = gn::register_fwd("wide_odd_fwd", /*prio=*/15, wide_odd_handles, wide_odd_run);
const bool wide_odd_bwd_registered = gn::register_bwd("wide_odd_bwd", /*prio=*/9, wide_odd_handles, wide_odd_run);

}  // namespace
