# TDN and AWM precision

Both the original implementation and the optimized implementation use FP32
intermediates with BF16 boundaries in BF16 training. The following measurements
use the same BF16-quantized inputs and an independent FP64 mathematical reference.
Errors are relative L2 distances to unrounded FP64 outputs; they include output
rounding but exclude input information already lost before these operators.

| Operation | Original BF16 output error | Optimized BF16 output error | Original fwd+bwd | Optimized fwd+bwd |
|---|---:|---:|---:|---:|
| TDN, B1 L8192 D1536 G24 | 0.166638% | 0.166630% | 1.2631 ms | 0.0891 ms |
| Full AWM, B1 L16384 H8 KV4 D256 C4096 | 0.339821% | 0.339905% | 2.4309 ms | 2.0042 ms |

The AWM table measures the complete working-memory component with nonzero output
and gradients, not only the chunk-softmax primitive. Original AWM explicitly
computes softmax in FP32 and casts probabilities back to the input dtype. Its
memory operations retain that dtype. Both complete AWM paths repeat bitwise in
this fixed-input check. Their direct BF16 output distance is 0.08706%; the
difference is reproducible arithmetic rounding, not random measurement noise.
The optimized AWM key-gradient error is 0.3919%, versus 0.4245% originally.

Retaining FP32 TDN outputs reduces output error to 0.00735% at 0.3398 ms, about
3.82x the BF16 cost. Casting the result back to BF16 costs 0.4014 ms and leaves
0.1668% error; it does not improve this BF16 output. The original CUB path uses
`float4` scans, float decay coefficients, and float running state even with FP64
input, so calling the production operator with `.double()` is not a strict FP64
reference. Both CUB and optimized paths are evaluated against separate FP64 math.

For full AWM, FP32 internal tensors with TF32 matmuls and BF16 input/output
boundaries reduce output error to 0.1688% at 2.0205 ms in this shape. Strict IEEE
FP32 internals with the same BF16 boundaries cost 3.8797 ms and yield 0.1659%
error. Retaining FP32 outputs instead gives 0.03129% error with TF32 and 0.000128%
with IEEE FP32. Promoted tensor storage doubles. These are isolated component
measurements; no per-component precision switch is integrated into the training
model, and the FP32 alternatives have not been qualified by a model training run.

The original and optimized BF16 paths have similar accuracy here. This does not
establish bitwise equivalence, eliminate systematic rounding differences, or
prove long-run convergence equivalence. See [gekko-training.md](gekko-training.md)
for a short model comparison and repeat controls.
