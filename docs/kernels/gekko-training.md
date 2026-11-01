# Gekko training comparison

The four optimized kernel families together reduce warm training-step latency from
737.41 ms to 624.76 ms (1.1803x) on B300 sm_103a. This is a combined model result;
it does not assign a speedup to any one family. Isolated kernel speedups in the
individual measurement tables have different scopes.

The stock `gekko-0.9B` preset has 1,651,440,640 parameters including embeddings.
Both implementations load one saved FP32 seed-42 initialization, use the same
12,800 paired per-rank batches across two placements, and train for 200 steps per
placement with fresh AdamW. Each arm uses four nodes / 32 GPUs; the two arms swap
node groups for the second placement. Training uses BF16, FP32 master weights,
16,384-token sequences, 4,096-token chunks, and global batch size 32. The data is
from public IFM/TxT360-v2, tokenized with the K2-Horizon tokenizer. Each run reads
104,857,600 training tokens. This is training from initialization, not continuation
of pretrained Gekko weights.

Forward and backward calls are recorded for group LayerNorm, group RMSNorm, TDN,
AWM chunk softmax, and RoPE. AWM key gradients are explicitly nonzero. Without
incoming memory, the first two AWM chunks have zero output, so an 8K / two-chunk
run would not establish effective AWM training coverage. Timing excludes the
first 20 steps, during which operation shapes and nonzero gradients are recorded,
and excludes checkpoint and evaluation time. The physical node-group speedups
are 1.1765x and 1.1841x. All 134 combined native integration tests pass.

Training is not bit-exact. The mean absolute paired loss difference is 0.004093,
with maximum 0.025235. Final held-out perplexities are 782.998 / 787.349 and
785.771 / 785.104 (original / optimized). Final FP32 weight relative L2 distances
are 0.6474% and 0.5316%. Repeating the original binary gives 0.4207% weight
distance; repeating the optimized binary gives 0.6696%. These controls show that
training variability is material, but two placements do not establish statistical
equivalence or prove that every implementation difference is noise. No consistent
perplexity regression appears across these two placements. Longer training and
pretrained-checkpoint continuation remain unqualified.

The numerical and timing measurements are in
[gekko-training-measurements.json](gekko-training-measurements.json).
