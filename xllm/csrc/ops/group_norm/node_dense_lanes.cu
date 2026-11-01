// Registers the dense-lane kernel pair (dense_lanes.cu): the many-group class where a group is 1..32 sixteen-byte
// vectors, mapped with W = V lanes a group rather than the next power of two, so groups of 3, 5 and 10 vectors keep
// every lane busy.
//
// The same entry points are registered under two conditions, which a single priority cannot express:
//   * the non-affine calls, forward and backward, before everything else (priority 26);
//   * the affine backward, only after the one-pass backward declines it (priority 10).
// The affine forward stays on the other forward kernels, which are faster on it.
#include "dense_lanes.h"  // xllm::ops::dense_lanes::{Eligible, Fwd, Bwd}
#include "dispatch.h"

namespace {

// The shape test of dense_lanes plus every tensor it indexes contiguous.
bool dense_lanes_ok(const gn::Shape& s, const gn::Call& c) {
  return xllm::ops::dense_lanes::Eligible(*c.x_or_y, s.groups) && c.dy->is_contiguous() && c.rstd->is_contiguous() &&
         (c.gamma->numel() == 0 || c.gamma->is_contiguous()) && (c.beta->numel() == 0 || c.beta->is_contiguous()) &&
         (c.mean->numel() <= 1 || c.mean->is_contiguous());
}

void run_dense_lanes(const gn::Shape& s, const gn::Call& c) {
  if (s.mode == gn::kFwd) {
    xllm::ops::dense_lanes::Fwd(*c.x_or_y, s.groups, s.norm == gn::kLayer, *c.gamma, *c.beta, c.eps, *c.y,
                                *c.mean_out, *c.rstd_out);
  } else {
    xllm::ops::dense_lanes::Bwd(*c.x_or_y, *c.dy, s.groups, s.norm == gn::kLayer, s.memory_efficient, *c.mean,
                                *c.rstd, *c.gamma, *c.beta, *c.dx, *c.dgamma, *c.dbeta);
  }
}

bool nonaffine_handles(const gn::Shape&) { return true; }
bool nonaffine_run(const gn::Shape& s, const gn::Call& c) {
  if (s.affine || !dense_lanes_ok(s, c)) return false;
  run_dense_lanes(s, c);
  return true;
}

bool affine_bwd_handles(const gn::Shape& s) { return s.mode == gn::kBwd && s.affine; }
bool affine_bwd_run(const gn::Shape& s, const gn::Call& c) {
  if (!dense_lanes_ok(s, c)) return false;
  run_dense_lanes(s, c);
  return true;
}

// Above everything for the non-affine class, where it is 12-13x faster than xllm's kernels on the many-group shapes;
// below the one-pass backward for the affine one.
const bool nonaffine_fwd_registered =
    gn::register_fwd("dense_lanes_nonaffine", /*prio=*/26, nonaffine_handles, nonaffine_run);
const bool nonaffine_bwd_registered =
    gn::register_bwd("dense_lanes_nonaffine", /*prio=*/26, nonaffine_handles, nonaffine_run);
const bool affine_bwd_registered =
    gn::register_bwd("dense_lanes_affine_bwd", /*prio=*/10, affine_bwd_handles, affine_bwd_run);

}  // namespace
