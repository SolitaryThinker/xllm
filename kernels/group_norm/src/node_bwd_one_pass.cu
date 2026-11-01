// Registers the one-pass backward of bwd_one_pass.cu: dx, dgamma and dbeta from one read of x_or_y and dy.
#include "bwd_one_pass.h"  // xllm::ops::GroupNormBwdOnePass
#include "dispatch.h"

namespace {

// The entry point answers false for every shape it does not vectorize, so this node claims the whole backward pass and
// declines in `run`.
bool bwd_one_pass_handles(const gn::Shape& s) { return s.mode == gn::kBwd; }

bool bwd_one_pass_run(const gn::Shape& s, const gn::Call& c) {
  return xllm::ops::GroupNormBwdOnePass(*c.dy, *c.x_or_y, *c.mean, *c.rstd, *c.gamma, *c.beta, s.channels, s.groups,
                                        s.norm == gn::kLayer, s.memory_efficient, *c.dx, *c.dgamma, *c.dbeta);
}

const bool bwd_one_pass_registered = gn::register_bwd("bwd_one_pass", /*prio=*/12, bwd_one_pass_handles, bwd_one_pass_run);

}  // namespace
