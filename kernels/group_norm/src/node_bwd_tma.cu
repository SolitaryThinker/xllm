// Registers the TMA backward of bwd_tma.cu: a persistent, warp-specialized kernel with a TMA load ring that computes dx
// and dgamma/dbeta in one pass.
#include "bwd_tma.h"
#include "dispatch.h"

namespace {

// The calls this node takes: those gn_tma::backward can run (bf16, whole 16-byte chunks a group, at most 8192
// channels, the modes it instantiates, a mapping for the group width) and that it runs faster than the nodes below it.
// gn_tma::backward still checks the tensors themselves (contiguity, 16-byte alignment, the statistics' layout, the
// shared-memory budget) and declines, having launched nothing, what it cannot run.
bool bwd_tma_handles(const gn::Shape& s) {
  if (s.mode != gn::kBwd || s.dtype != at::kBFloat16) return false;
  if (s.groups <= 0 || s.cpg <= 0 || s.cpg % 8 != 0 || s.channels > 8192) return false;
  const bool layer = s.norm == gn::kLayer;
  const bool rms_affine_x = !layer && s.affine && !s.memory_efficient;
  const bool ln_affine_x = layer && s.affine && !s.memory_efficient;
  // gn_tma::backward also runs RMS without gamma from y, but the dense-lane kernel is faster on every such shape, so
  // this node does not claim it.
  if (!rms_affine_x && !ln_affine_x) return false;
  const int64_t V = s.cpg / 8;  // 16-byte chunks a group
  if (V > 32) {
    // Wide: one warp a group, Q = ceil(V / 32) chunks a lane in 4..6, at most 8 groups, RMS with gamma from x.
    if (!rms_affine_x) return false;
    const int64_t Q = (V + 31) / 32;
    if (Q >= 4 && Q <= 6 && s.groups <= 8) return true;
    // Split: a group of 128 chunks or more over up to 16 warps of four chunks a lane.
    return V >= 128 && s.groups * ((V + 127) / 128) <= 16;
  }
  const int64_t GW = 32 / V;                        // whole groups a warp-chunk
  if ((s.groups + GW - 1) / GW > 32) return false;  // at most 32 warp-chunks a row
  // The layer norm and RMS with gamma, every narrow group count: tiles are handed out on demand, so the node is ahead of
  // the one-pass backward at 64 groups too.
  return true;
}

bool bwd_tma_run(const gn::Shape& s, const gn::Call& c) {
  if (!c.x_or_y || !c.dy || !c.mean || !c.rstd || !c.gamma || !c.beta || !c.dx || !c.dgamma || !c.dbeta) return false;
  return gn_tma::backward(s.norm == gn::kLayer, s.groups, s.memory_efficient, *c.x_or_y, *c.dy, *c.mean, *c.rstd,
                          *c.gamma, *c.beta, *c.dx, *c.dgamma, *c.dbeta);
}

// Priority 27: above the nodes whose shapes it takes (by-width 24, wide/odd 9, one-pass 12) and alone at its priority.
// It claims no non-affine call, so it never meets the dense-lane non-affine node (26).
const bool bwd_tma_registered = gn::register_bwd("bwd_tma", 27, bwd_tma_handles, bwd_tma_run);

}  // namespace
