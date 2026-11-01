// Registers the forward kernel of fwd_lane_groups.cu (a group is a few 16-byte lanes of a warp) with the dispatcher.
#include "dispatch.h"
#include "fwd_lane_groups.h"  // xllm::ops::GroupNormFwdLaneGroups

namespace {

// The entry point does its own shape checks and returns false for anything it will not take, so this claims the whole
// forward pass and declines in `run`: returning false hands the call to the next node, and finally to xllm's kernels.
bool fwd_lane_groups_handles(const gn::Shape& s) { return s.mode == gn::kFwd; }

bool fwd_lane_groups_run(const gn::Shape& s, const gn::Call& c) {
  // The entry point takes `.contiguous()` copies of its outputs, which would silently drop the result of a
  // non-contiguous output; outputs are always allocated fresh by the caller, and are contiguous.
  return xllm::ops::GroupNormFwdLaneGroups(*c.x_or_y, *c.gamma, *c.beta, s.groups, c.eps, s.norm == gn::kLayer, *c.y,
                                           *c.mean_out, *c.rstd_out);
}

const bool fwd_lane_groups_registered =
    gn::register_fwd("fwd_lane_groups", /*prio=*/20, fwd_lane_groups_handles, fwd_lane_groups_run);

}  // namespace
