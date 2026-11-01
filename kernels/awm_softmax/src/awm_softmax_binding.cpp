// Entry point of the chunked softmax and logsumexp over time of xllm's Gekko adaptive working memory: one call of one
// operation, chosen by mode. It is called positionally with the inputs, then the destinations: kkey, lse_out,
// prev_kkey, dkey. An output a mode does not produce is a one-element placeholder that stays as given; an input it
// does not take is an empty tensor. Every tensor is contiguous.
//
//   mode 0 (forward): key [B, L, K, S], key_mask [B, L] or empty, prev_key [B, C, K, S] or empty
//                     -> kkey [B, K, S, L], lse_out [B, K, S, N] (slot n >= 1: chunk n - 1; slot 0: prev_key's, or
//                        left as given), prev_kkey [B, K, S, C] with prev_key
//   mode 1 (backward): key, key_mask, lse [B, K, S, N], dkkey [B, K, S, L - C], dlse [B, K, S, N] -> dkey [B, L, K, S]
//   mode 2 (decode, update_memory): key [B, K, C, S] -> kkey [B, K, S, C], lse_out [B, K, S]
//
// The kernels are in awm_chunk_softmax.cu (forward, decode, and a fused backward) and awm_two_pass_bwd.cu (backward):
// one CTA per (row, kv head, channel block of ST, chunk), the chunk's tile transposed once through shared memory.
// Shapes the fast path cannot take (see pick_sp, pick_st and awm_bwd_try) fall back to xllm's own ATen operations
// (fwd_ref, bwd_ref, decode_ref), which are the definition of what is computed.
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

#include <limits>

#include "awm_chunk_softmax.h"

namespace {

constexpr int64_t kFwd = 0;
constexpr int64_t kBwd = 1;
constexpr int64_t kDecode = 2;

// Reference path: xllm 692bc25's computation as it runs, the same ATen operations in the same order.
at::Tensor masked_time(const at::Tensor& key, const at::Tensor& key_mask) {
  const int64_t B = key.size(0), L = key.size(1);
  auto k = key.permute({0, 2, 3, 1}).to(at::kFloat).contiguous();
  if (key_mask.numel() > 0) {
    k.masked_fill_(key_mask.view({B, 1, 1, L}), -std::numeric_limits<float>::infinity());
  }
  return k;
}

void decode_ref(const at::Tensor& key, at::Tensor kkey, at::Tensor lse_out) {
  auto k = key.to(at::kFloat).transpose(2, 3).contiguous();
  kkey.copy_(at::softmax(k, -1));
  at::logsumexp_out(lse_out, k, {-1}, false);
}

void fwd_ref(const at::Tensor& key, const at::Tensor& key_mask, const at::Tensor& prev_key, at::Tensor kkey,
             at::Tensor lse_out, at::Tensor prev_kkey, int64_t C) {
  const int64_t B = key.size(0), L = key.size(1), K = key.size(2), S = key.size(3);
  const int64_t N = L / C;
  auto k = masked_time(key, key_mask).view({B, K, S, N, C});
  kkey.view({B, K, S, N, C}).copy_(at::softmax(k, -1));
  if (N > 1) {
    auto lse_chunks = lse_out.narrow(3, 1, N - 1);
    at::logsumexp_out(lse_chunks, k.narrow(3, 0, N - 1), {-1}, false);
  }
  if (prev_key.numel() > 0) {
    auto p = prev_key.permute({0, 2, 3, 1}).to(at::kFloat).contiguous();
    prev_kkey.copy_(at::softmax(p, -1));
    auto slot0 = lse_out.select(3, 0);
    at::logsumexp_out(slot0, p, {-1}, false);
  }
}

void bwd_ref(const at::Tensor& key, const at::Tensor& key_mask, const at::Tensor& lse, const at::Tensor& dkkey,
             const at::Tensor& dlse, at::Tensor dkey, int64_t C) {
  const int64_t B = key.size(0), L = key.size(1), K = key.size(2), S = key.size(3);
  const int64_t N = L / C;
  const auto dtype = key.scalar_type();
  auto k = masked_time(key, key_mask).view({B, K, S, N, C});
  auto kk = at::softmax(k, -1);
  auto kg = dkey.permute({0, 2, 1, 3});
  for (int64_t c = N - 1; c >= 1; --c) {
    auto g = dkkey.narrow(3, (c - 1) * C, C).to(at::kFloat);
    auto sb = at::_softmax_backward_data(g, kk.select(3, c - 1), -1, at::kFloat).to(dtype);
    kg.narrow(2, (c - 1) * C, C).copy_(sb.transpose(2, 3));
  }
  if (N > 1) {
    auto e = at::exp(k.narrow(3, 0, N - 1) - lse.narrow(3, 1, N - 1).unsqueeze(4));
    auto t = at::mul(dlse.narrow(3, 1, N - 1).unsqueeze(4), e).to(dtype);
    kg.narrow(2, 0, L - C).add_(t.view({B, K, S, (N - 1) * C}).transpose(2, 3));
  }
  if (key_mask.numel() > 0) {
    kg.masked_fill_(key_mask.view({B, 1, L, 1}), 0);
  }
}

// The channel block of the fused one-pass backward kernel (bwd_kernel), which serves the shapes the two-pass path
// declines, namely a channel count that is not a multiple of 16: ST = 16 channels would be one 32 B sector per row of
// key. C must be a multiple of 32*VEC, one reduce/write round (256 for a 2-byte key, 128 for a 4-byte one), so that no
// reduce round is truncated. It need not be a multiple of RPI, the steps one load round covers: load_tile covers the
// steps its rounds do not reach with one guarded row per thread (its nTail). A 4-byte key needs nothing more: RPI is
// 128 (ST = 16) or 256 (ST = 8) there, so the `C % 256` below already implies it.
//
// Two conditions cannot move: the `C % 256` this function starts with and the tile-size test. Both are
// truncation/fit guards of the kernel; everything else is a choice between two correct kernels.
int pick_st(int64_t B, int64_t K, int64_t C, int64_t S, int64_t esz, int64_t N) {
  if (C % 256 != 0) return 0;
  auto ok = [&](int64_t st) {
    if (S % st != 0) return false;
    if ((int64_t)(C + 8) * st * esz > (int64_t)awm::kMaxShared - 1024) return false;
    return true;
  };
  const bool ok16 = ok(16), ok8 = ok(8);
  if (!ok16 && !ok8) return 0;
  // ST = 8 halves the tile (C * 16 * 2 = 128 KiB at C = 4096, so 2-3 CTAs per SM instead of 1-2) and halves the
  // read run of a key row (16 B instead of 32 B, the neighbouring channel block sharing the sector through L2). Two
  // things decide it, both in bytes and CTAs rather than by regime:
  //   * a tile of 128 KiB is worth halving (C >= 4096): at C = 4096 the narrow tile is 1.10-1.23x faster on the
  //     backward, and at C = 1024-2048 the wide one is 1.25-1.33x faster;
  //   * a grid that leaves the machine unfilled even at ST = 16 (n16 <= 64 CTAs) wants more CTAs, not a bigger
  //     tile: 1.06-1.29x at C = 1024-2048, N <= 2, on the backward.
  const int64_t n16 = B * K * (S / 16) * (N - 1);
  const bool narrow = (C >= 4096) || (n16 <= 64);
  if (!narrow) return ok16 ? 16 : 8;
  return ok8 ? 8 : 16;
}

// The forward's (and decode's) channel block ST and cluster split P. A wider read run buys read rate, but little. On
// a read-only stream of 256 MiB each way at grid 4096, reading 16 rows per warp instruction (the 16-channel block)
// takes 54.27 us against 48.06 for a coalesced read (+12.9%); 8 rows 52.19 (+8.6%); 4 rows 52.26 (+8.7%); 2 rows
// 48.13 (+0.1%). Read those as a curve: the ST = 16 -> ST = 32 step is the 16 -> 8 rows step and buys 3.99%, the next
// ST the table carries, 64 (4 rows), buys nothing, and only the 2-row read reaches coalesced, which is an ST = 128
// that kSp does not hold. At fixed grid and volume a read+write stream reaches 4609 GB/s with 16 channels, 5671 with
// 32 and 6167 with 64. A wide block does not pay on every stream: (8,1), the narrowest block, is what the masked
// stream runs.
// The wide block is not free: (32,4) costs +11.24% on prefill_4x256_b8_l16384 (0.1530 -> 0.1702 ms) against a
// +3.99% ceiling on one phase of the same call. The single-CTA alternative is refused by sp_ok's own
// two-CTAs-per-SM shared cap (its 68 KiB arm): (32,2)'s tile is 132 112 B, which fits kMaxShared (200 KiB = 204 800)
// but only at one CTA per SM, and one CTA per SM is expensive. The tile grows with ST and the cluster grows with the
// tile, so width and depth are inseparable on this interface; reading 4 rows per warp instruction without growing
// the tile would need a different shared layout. What the table below carries: (16,2) for the unmasked shapes, (8,1)
// for the masked ones, (32,4) and (16,1) as fallbacks.
// A wide block needs a small tile to hold two CTAs per SM, and the tile is what the split buys:
// one rank holds ST x C/P. Both loops have a grain -- RPI steps per load round, 32*VEC per reduce and write round --
// and the guard is stated in those, not in C.
struct Sp {
  int st, p;
};

// The split, chosen where it wins: the unmasked forward and decode at C >= 4096. Everywhere else the forward keeps
// single-CTA cells. The masked stream takes (8,1) (the split loses on masked shapes: b4_l8192 34.6 us at (8,1) against
// 44.5 at (32,4)), and the single-chunk prev call takes (16,1) (13.3 against 13.7). The one cell not in this table is
// the long-chunk band's, a last resort rather than a preference: whatever `sp_ok` accepts at the narrow width, only
// after everything here and both P = 1 cells have declined (see pick_sp).
static const Sp kSp[] = {{16, 2}, {32, 4}};
constexpr int kSpN = 2;

// The launcher's shared-memory formula and thread count (`launch_fwd` in awm_chunk_softmax.cu), repeated here because
// the guard has to answer before a kernel is instantiated; keep the three in step: `nt` is `launch_fwd<T, ST, NT, P>`'s
// NT (512 in every cell `launch_fwd_t` dispatches), and `sh` is its `ST*(C/P + PAD)*sizeof(T) + 2*P*ST*4 + 16`.
bool sp_ok(int64_t C, int64_t S, int64_t esz, int st, int p) {
  const int64_t vec = 16 / esz;
  if (st % vec != 0) return false;
  if (S % st != 0) return false;
  if (C % p != 0) return false;
  const int64_t nt = 512;
  const int64_t rpi = nt / (st / vec);
  const int64_t sl = C / p;
  // The load's round grain (`sl % rpi`) is not a guard: load_tile covers the steps its rounds do not reach with one
  // guarded row per thread (its nTail), and the kernel passes it `SL - nIter * RPI`. Requiring the grain here would
  // refuse a 2-byte key at C = 6400, ST = 8 (rpi = 512, 6400 % 512 = 256) in every cell and send the call to the
  // ATen path. The reduce/write round cannot move: the reduce and the write index the tile by whole 32*VEC rounds.
  if (rpi < 1 || sl % (32 * vec) != 0) return false;
  const int64_t sh = st * (sl + 8) * esz + 2 * p * st * 4 + 16;
  if (sh > (int64_t)awm::kMaxShared - 1024) return false;
  // A split is only worth taking while its tile keeps two CTAs per SM by shared memory (a 66 064 B tile fits three of
  // the ~227 KiB, so this guard is slack); what holds the kernel at 2 CTAs/SM is the register file (62 regs x 512
  // threads). The cap binds every cell it applies to, and it is lifted for exactly one cell: the narrow width at two
  // ranks of a 4-byte key. That cell exists because of a hard limit, not the cap: above C ~ 6357 neither whole-chunk
  // cell fits shared memory at all (32C + 336 > 203 776, and `(16,1)` is bigger still), and at every C where this cell
  // fires the band's `(8,4)`/`(8,8)` fail the grain or the cap (see pick_sp). It runs at 1 CTA/SM by design, 102 800 B
  // of tile at C = 6400 up to 201 104 B at 12 544; the alternative is the ATen path. The table's `(16,2)`/`(32,4)` and
  // the band's `(8,4)`/`(8,8)` keep the cap.
  if (p > 1 && sh > (((esz == 4 && st == 8 && p == 2) ? (int64_t)awm::kMaxShared - 1024 : 68 * 1024))) return false;
  return true;
}

// `n16` is the grid of a 16-channel single-CTA cell: B*K*(S/16)*N, the CTA count it would launch. `mode` is needed
// because decode and a single-chunk prefill call want different cells at the same C.
bool pick_sp(int64_t mode, bool masked, bool has_prev, int64_t C, int64_t N, int64_t n16, int64_t S, int64_t esz,
             int& ST, int& P) {
  const bool narrow = (C >= 4096) || (n16 <= 64);
  // The default predicate, for every shape the split does not take. The cell depends on the grid as well as on C: a
  // list that ignores C and n16 would send the masked forward to 8 at C = 2048, where 16 is ahead by 1.25-1.33x.
  //   masked forward, decode: narrow (C >= 4096 or a grid under ~64 CTAs) -> 8, else 16;
  //   unmasked forward: the whole key without a mask prefers 16, the prev-chunk call at N > 1 prefers 8 (34.1 vs
  //   36.6 us at C = 4096), and the single-chunk prev call prefers 16 (13.3 vs 14.9).
  int st = (masked || mode == kDecode) ? (narrow ? 8 : 16)
                                      : ((!has_prev || N == 1) ? 16 : (narrow ? 8 : 16));
  const bool wantSplit = !masked && C >= 4096 && !(mode == kFwd && has_prev && N == 1);
  if (wantSplit) {
    for (int i = 0; i < kSpN; ++i) {
      if (sp_ok(C, S, esz, kSp[i].st, kSp[i].p)) {
        ST = kSp[i].st;
        P = kSp[i].p;
        return true;
      }
    }
  }
  P = 1;
  const int alt = (st == 16) ? 8 : 16;
  ST = sp_ok(C, S, esz, st, 1) ? st : (sp_ok(C, S, esz, alt, 1) ? alt : 0);
  if (ST > 0) return true;
  // The band: every cell above holds the whole chunk in one CTA's tile, and a 4-byte element at a long chunk fits none
  // of them (fp32, C = 8192, 8 channels: 8 x 8200 x 4 = 262 KiB against 200 KiB; 16 channels is worse). A split is the
  // only path whose tile comes back under budget, and `sp_ok` already evaluates it on the slice (`sl = C/p`), so the
  // one thing missing is a candidate at the narrow width.
  //
  // The masked route is open on a split too. What keeps it correct is the clamp in `fwd_kernel`'s prefix skip (a rank
  // whose whole slice is -inf would otherwise zero-fill past the end of kkey), not a dispatch preference.
  //
  // P = 2 is deliberately not a candidate for a 2-byte key. The split's tile cap (68 KiB) is 2.94x tighter than the
  // P = 1 budget (200 KiB) while a two-way split is only 2x, so every C where a P = 2 slice fits is a C that
  // sp_ok(8,1) and sp_ok(16,1) have already answered; the loop would carry a first candidate that can never fire.
  // Start at 4.
  //
  // What this covers, as a rule: at (8,4), C a multiple of 512 in 6656..8192 (sl = C/4, a multiple of 128, tile
  // 53.6-66.1 KiB); at (8,8), C a multiple of 1024 in 9216..16384 (tile <= 66.1 KiB), and a 2-byte key above that.
  // Every other multiple of 256 falls to the 4-byte loop below, or to the ATen path: 6400, 6912, 7424, 7936 (not
  // multiples of 512, so sl is not a multiple of 128 - answered at P = 2 below), 8448, 8704 (sl = 2176 is 70 160 B at
  // P = 4, over the cap; 1088 at P = 8 fails the grain), 8960, 9472, 9728, 9984, 10496, 10752, 11008, 11520, 11776,
  // 12032, 12544 (not multiples of 1024 in the P = 8 band, or over its cap), and everything above 16384.
  static const int kPBand[2] = {4, 8};
  for (int i = 0; i < 2; ++i) {
    if (sp_ok(C, S, esz, 8, kPBand[i])) {
      ST = 8;
      P = kPBand[i];
      return true;
    }
  }
  // A 4-byte element is the case the paragraph above excludes: no whole-chunk tile fits (this block runs only at
  // ST == 0), and nothing under the split cap passes the grain until P = 2, where sl is half a multiple of 256, a
  // multiple of 128, which is exactly 32*VEC for a 4-byte element. So (8,2) is grain-legal at every multiple of 256,
  // and the only thing in its way is the cap, which `sp_ok` lifts for this one cell (`esz == 4 && st == 8 && p == 2`)
  // and no other. Its reach, from the guard's own arithmetic:
  //   8*(C/2 + 8)*4 + 2*2*8*4 + 16 = 16C + 400 <= kMaxShared - 1024 = 203 776  <=>  C <= 12 711,
  // i.e. every multiple of 256 up to 12 544 that no cell above claims: 6400, 6912, 7424, 7936 (the grain holes of
  // (8,4) and (8,8)), 8448, 8704, 8960, 9472, 9728, 9984, 10496, 10752, 11008, 11520, 11776, 12032, 12544. What kPBand
  // takes keeps kPBand: (8,4) at the multiples of 512 in 6656..8192, and (8,8) at the multiples of 1024 in
  // 9216..16384 (sl = C/8 is a multiple of 128 there; tile 37 648-66 320 B). Only what no cell answers - above
  // 16 384, or above 12 544 at a value that is not a multiple of 1024 - goes to the ATen path.
  // The grain holds at every C this cell takes by construction, which is why it is a single candidate with no ladder
  // under it. Covered is not fast: the slice keeps two CTAs per SM by shared memory only up to C = 7 239
  // (16C + 400 <= 116 224), so of the cells this one takes only 6400 and 6912 run two CTAs per SM; the rest run at
  // one. It takes 9.1 us against the ATen path's 54.4 us at B1 K1 S64, and 30.2 us against 76.8 at K4 x S256.
  if (ST == 0 && esz == 4 && sp_ok(C, S, esz, 8, 2)) {
    ST = 8;
    P = 2;
    return true;
  }
  return false;
}

int dtype_code(const at::Tensor& t) {
  switch (t.scalar_type()) {
    case at::kBFloat16: return 0;
    case at::kHalf: return 1;
    case at::kFloat: return 2;
    default: return -1;
  }
}

}  // namespace

// mode 1's kernel (awm_two_pass_bwd.cu): dkey from key, key_mask, lse, dkkey and dlse, two streaming passes over the bytes.
bool awm_bwd_try(const at::Tensor& key, const at::Tensor& key_mask, const at::Tensor& lse, const at::Tensor& dkkey,
                 const at::Tensor& dlse, at::Tensor& dkey, int64_t chunk_size);

void run(int64_t mode, int64_t chunk_size, const at::Tensor& key, const at::Tensor& key_mask,
         const at::Tensor& prev_key, const at::Tensor& lse, const at::Tensor& dkkey, const at::Tensor& dlse,
         at::Tensor kkey, at::Tensor lse_out, at::Tensor prev_kkey, at::Tensor dkey) {
  TORCH_CHECK(key.is_cuda() && key.dim() == 4 && key.is_contiguous(), "key: a contiguous 4-d tensor on the GPU");
  // Run on the device the tensors live on, not on whichever device the caller happens to have current: every ATen op
  // installs this guard from its own inputs, and this entry point takes bare tensors and launches by hand, so it would
  // otherwise launch on the ambient device against foreign pointers (an illegal address, not a wrong answer). It is
  // also what makes getCurrentCUDAStream() below the stream of the tensor's device. One host-side line, before the
  // first launch, and it restores the caller's device on the way out.
  const c10::cuda::CUDAGuard device_guard(key.device());
  const int64_t C = chunk_size;
  const auto dtype = key.scalar_type();
  cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
  const int dc = dtype_code(key);
  const int64_t esz = key.element_size();

  if (mode == kDecode) {
    // key [B, K, C, S] -> kkey [B, K, S, C], lse_out [B, K, S]
    const int64_t B = key.size(0), K = key.size(1), Ce = key.size(2), S = key.size(3);
    int ST = 0, P = 1;
    const int64_t n16d = B * K * (S / 16);
    if (dc < 0 || !pick_sp(kDecode, false, false, Ce, 1, n16d, S, esz, ST, P) || Ce != C) {
      decode_ref(key, kkey, lse_out);
      return;
    }
    awm::FwdArgs a{};
    a.key = key.data_ptr();
    a.mask = nullptr;
    a.kout = kkey.data_ptr();
    a.lse = lse_out.data_ptr<float>();
    a.inBStr = K * Ce * S;
    a.inTStr = S;
    a.inKStr = Ce * S;
    a.outBStr = K * S * Ce;
    a.outKStr = S * Ce;
    a.outL = Ce;
    a.maskBStr = 0;
    a.lseBStr = K * S;
    a.lseKStr = S;
    a.lseNStr = 1;
    a.B = (int)B;
    a.K = (int)K;
    a.nbChan = (int)(S / ST);
    a.C = (int)Ce;
    a.cMul = (int)Ce;
    a.lseMaxC = 0;
    a.slotBase = 0;
    a.nTasks = (int)(B * K * (S / ST));
    if (awm::fwd_launch(a, a, 0, dc, ST, P, stream) != 0) decode_ref(key, kkey, lse_out);
    return;
  }

  const int64_t B = key.size(0), L = key.size(1), K = key.size(2), S = key.size(3);
  TORCH_CHECK(C > 0 && L % C == 0, "the length must be a multiple of chunk_size, as xllm asserts");
  const int64_t N = L / C;

  if (mode == kFwd) {
    int ST = 0, P = 1;
    const int64_t n16 = B * K * (S / 16) * N;
    if (dc < 0 || !pick_sp(kFwd, key_mask.numel() > 0, prev_key.numel() > 0, C, N, n16, S, esz, ST, P)) {
      fwd_ref(key, key_mask, prev_key, kkey, lse_out, prev_kkey, C);
      return;
    }
    awm::FwdArgs a{};
    a.key = key.data_ptr();
    a.mask = key_mask.numel() > 0 ? key_mask.data_ptr() : nullptr;
    a.kout = kkey.data_ptr();
    a.lse = lse_out.data_ptr<float>();
    a.inBStr = L * K * S;
    a.inTStr = K * S;
    a.inKStr = S;
    a.outBStr = K * S * L;
    a.outKStr = S * L;
    a.outL = L;
    a.maskBStr = L;
    a.lseBStr = K * S * N;
    a.lseKStr = S * N;
    a.lseNStr = N;
    a.B = (int)B;
    a.K = (int)K;
    a.nbChan = (int)(S / ST);
    a.C = (int)C;
    a.cMul = (int)C;
    a.lseMaxC = (int)(N - 2);
    a.slotBase = 1;
    a.nTasks = (int)(B * K * (S / ST) * N);

    awm::FwdArgs p{};
    const bool hasPrev = prev_key.numel() > 0;
    if (hasPrev) {
      p.key = prev_key.data_ptr();
      p.mask = nullptr;
      p.kout = prev_kkey.data_ptr();
      p.lse = lse_out.data_ptr<float>();
      p.inBStr = C * K * S;
      p.inTStr = K * S;
      p.inKStr = S;
      p.outBStr = K * S * C;
      p.outKStr = S * C;
      p.outL = C;
      p.maskBStr = 0;
      p.lseBStr = K * S * N;
      p.lseKStr = S * N;
      p.lseNStr = N;
      p.B = (int)B;
      p.K = (int)K;
      p.nbChan = (int)(S / ST);
      p.C = (int)C;
      p.cMul = (int)C;
      p.lseMaxC = 0;
      p.slotBase = 0;
      p.nTasks = (int)(B * K * (S / ST));
    }
    if (awm::fwd_launch(a, hasPrev ? p : a, hasPrev ? 1 : 0, dc, ST, P, stream) != 0) {
      fwd_ref(key, key_mask, prev_key, kkey, lse_out, prev_kkey, C);
    }
    return;
  }

  // mode 1 (awm_two_pass_bwd.cu): dkey from key, key_mask, lse, dkkey and dlse by two streaming passes with no
  // cross-CTA barrier, the dot of each chunk carried between them through a scratch. It takes bf16, fp16 and fp32 keys
  // (its geometry follows the element size); the one-pass kernel below takes a channel count that is not a multiple
  // of 16, and any dtype it does not name goes to the ATen path.
  if (mode == kBwd && awm_bwd_try(key, key_mask, lse, dkkey, dlse, dkey, C)) return;
  TORCH_CHECK(mode == kBwd, "mode is 0 (forward), 1 (backward) or 2 (decode)");
  // With N < 2 there is nothing to compute: the reference's dkey is all zero there (chunks 0..N-2 are empty), and the
  // destination arrives zeroed, as the write contract of the backward pass requires of the caller. A memset here
  // would be a no-op launch on every call and would hide the assumption instead of stating it.
  if (N < 2) return;
  const int ST = pick_st(B, K, C, S, esz, N);
  if (dc < 0 || ST == 0) {
    bwd_ref(key, key_mask, lse, dkkey, dlse, dkey, C);
    return;
  }
  awm::BwdArgs a{};
  a.key = key.data_ptr();
  a.mask = key_mask.numel() > 0 ? key_mask.data_ptr() : nullptr;
  a.lse = lse.data_ptr<float>();
  a.dkkey = dkkey.data_ptr();
  a.dlse = dlse.data_ptr<float>();
  a.dkey = dkey.data_ptr();
  a.inBStr = L * K * S;
  a.inTStr = K * S;
  a.inKStr = S;
  a.maskBStr = L;
  a.lseBStr = K * S * N;
  a.lseKStr = S * N;
  a.lseNStr = N;
  a.dkCL = L - C;
  a.dkKStr = S;
  a.B = (int)B;
  a.K = (int)K;
  a.nbChan = (int)(S / ST);
  a.C = (int)C;
  a.N = (int)N;
  a.nTasks = (int)(B * K * (S / ST) * (N - 1));
  if (awm::bwd_launch(a, dc, ST, stream) != 0) bwd_ref(key, key_mask, lse, dkkey, dlse, dkey, C);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("run", &run); }
