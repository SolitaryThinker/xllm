// The registry and the fallback to xllm's kernels declared in dispatch.h. See that header for what this is for.
#include "dispatch.h"

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "group_layer_norm.h"
#include "group_rms_norm.h"

namespace gn {
namespace {

struct Entry {
  const char* name;
  int prio;
  HandleFn handles;
  RunFn run;
};

// Function-local statics: safe from any translation unit's static initialisation, whichever runs first.
std::vector<Entry>& fwd_table() {
  static std::vector<Entry> t;
  return t;
}
std::vector<Entry>& bwd_table() {
  static std::vector<Entry> t;
  return t;
}

// GN_IMPL is read once. "off" sends every call to xllm's kernels; "fwd:name,bwd:name" restricts the candidates to one
// implementation per direction.
struct Override {
  bool off = false;
  std::string fwd;
  std::string bwd;
};

const Override& override_() {
  static Override o = [] {
    Override r;
    const char* e = std::getenv("GN_IMPL");
    if (e == nullptr) return r;
    if (std::strcmp(e, "off") == 0) {
      r.off = true;
      return r;
    }
    std::string s(e);
    size_t pos = 0;
    while (pos <= s.size()) {
      size_t comma = s.find(',', pos);
      std::string item = s.substr(pos, comma == std::string::npos ? std::string::npos : comma - pos);
      size_t colon = item.find(':');
      if (colon != std::string::npos) {
        std::string key = item.substr(0, colon);
        std::string val = item.substr(colon + 1);
        if (key == "fwd") r.fwd = val;
        else if (key == "bwd") r.bwd = val;
      } else if (item == "off") {
        r.off = true;
      }
      if (comma == std::string::npos) break;
      pos = comma + 1;
    }
    return r;
  }();
  return o;
}

// The nodes that claim this shape, in the order they are tried: a node that declines at run time hands the call to the
// next one, and xllm's kernels take whatever is left. Ties are broken by name, so the order never depends on link
// order.
std::vector<const Entry*> candidates(const std::vector<Entry>& t, const Shape& s, const std::string& forced) {
  std::vector<const Entry*> v;
  for (const Entry& e : t) {
    if (!forced.empty() && forced != e.name) continue;
    if (!e.handles(s)) continue;
    v.push_back(&e);
  }
  std::sort(v.begin(), v.end(), [](const Entry* a, const Entry* b) {
    const int pa = a->prio;
    const int pb = b->prio;
    if (pa != pb) return pa > pb;
    return std::strcmp(a->name, b->name) < 0;
  });
  return v;
}

bool reg(std::vector<Entry>& t, const char* name, int prio, HandleFn handles, RunFn run) {
  if (override_().off) return false;
  for (const Entry& e : t) {
    if (std::strcmp(e.name, name) == 0) return false;  // registered twice: keep the first
  }
  t.push_back(Entry{name, prio, handles, run});
  return true;
}

}  // namespace

bool register_fwd(const char* name, int prio, HandleFn handles, RunFn run) {
  return reg(fwd_table(), name, prio, handles, run);
}
bool register_bwd(const char* name, int prio, HandleFn handles, RunFn run) {
  return reg(bwd_table(), name, prio, handles, run);
}

void forward(const Shape& s, const Call& c) {
  const Override& o = override_();
  if (!o.off) {
    for (const Entry* e : candidates(fwd_table(), s, o.fwd)) {
      if (e->run(s, c)) return;  // a false return is a decline: the next node is tried
    }
  }
  xllm_forward(s, c);
}

void backward(const Shape& s, const Call& c) {
  const Override& o = override_();
  if (!o.off) {
    for (const Entry* e : candidates(bwd_table(), s, o.bwd)) {
      if (e->run(s, c)) return;
    }
  }
  xllm_backward(s, c);
}

// --- xllm's own kernels (xllm 692bc25), the correct path for every shape xllm can call. ---
void xllm_forward(const Shape& s, const Call& c) {
  using namespace xllm::ops;
  if (s.mode == kFwd) {
    if (s.norm == kRMS) {
      if (s.affine) GroupRMSNormCUDAFwdAffine(*c.x_or_y, s.channels, s.groups, *c.gamma, c.eps, *c.y, *c.rstd_out);
      else GroupRMSNormCUDAFwd(*c.x_or_y, s.channels, s.groups, c.eps, *c.y, *c.rstd_out);
    } else {
      if (s.affine)
        GroupLayerNormCUDAFwdAffine(*c.x_or_y, s.channels, s.groups, *c.gamma, *c.beta, c.eps, *c.y, *c.mean_out,
                                    *c.rstd_out);
      else GroupLayerNormCUDAFwd(*c.x_or_y, s.channels, s.groups, c.eps, *c.y, *c.mean_out, *c.rstd_out);
    }
  }
}

void xllm_backward(const Shape& s, const Call& c) {
  using namespace xllm::ops;
  if (s.mode == kBwd) {
    if (s.norm == kRMS) {
      if (s.affine)
        GroupRMSNormCUDABwdAffine(*c.dy, *c.x_or_y, s.channels, s.groups, *c.rstd, *c.gamma, s.memory_efficient, *c.dx,
                                  *c.dgamma);
      else
        GroupRMSNormCUDABwd(*c.dy, *c.x_or_y, s.channels, s.groups, *c.rstd, s.memory_efficient, *c.dx);
    } else {
      if (s.affine)
        GroupLayerNormCUDABwdAffine(*c.dy, *c.x_or_y, s.channels, s.groups, *c.mean, *c.rstd, *c.gamma, *c.beta,
                                    s.memory_efficient, *c.dx, *c.dgamma, *c.dbeta);
      else
        GroupLayerNormCUDABwd(*c.dy, *c.x_or_y, s.channels, s.groups, *c.mean, *c.rstd, s.memory_efficient, *c.dx);
    }
  }
}

}  // namespace gn
