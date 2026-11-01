// Registers the forward (fwd_by_width.cu) and backward (bwd_by_width.cu) kernels whose mapping is chosen by channels
// per group. Each entry point checks its own shape class and returns false where it does not take the call, so the
// node passes its answer straight to the dispatcher.
#include "bwd_by_width.h"
#include "dispatch.h"
#include "fwd_by_width.h"

namespace {

bool by_width_handles(const gn::Shape& s) { return s.mode == gn::kFwd || s.mode == gn::kBwd; }

bool fwd_by_width_run(const gn::Shape& s, const gn::Call& c) {
  if (s.mode != gn::kFwd) return false;
  return gn::GroupNormFwdByWidth(s.norm == gn::kLayer, *c.x_or_y, s.groups, c.eps, *c.gamma, *c.beta, *c.y,
                                 *c.mean_out, *c.rstd_out);
}

bool bwd_by_width_run(const gn::Shape& s, const gn::Call& c) {
  if (s.mode != gn::kBwd) return false;
  return gn::GroupNormBwdByWidth(s.norm == gn::kLayer, *c.x_or_y, *c.dy, *c.mean, *c.rstd, *c.gamma, *c.beta,
                                 s.memory_efficient, *c.dx, *c.dgamma, *c.dbeta);
}

// Priority 24 in both directions: above the lane-group forward (20) and the wide/odd kernels (15), below the dense-lane
// kernels for non-affine calls (26).
const bool fwd_by_width_registered = gn::register_fwd("fwd_by_width", /*prio=*/24, by_width_handles, fwd_by_width_run);
const bool bwd_by_width_registered = gn::register_bwd("bwd_by_width", /*prio=*/24, by_width_handles, bwd_by_width_run);

}  // namespace
