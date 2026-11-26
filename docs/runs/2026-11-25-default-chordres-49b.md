# Default ChordRes, full 49.152B-token schedule

Use the admitted `chordres_update_soft_force` configuration with fixed zero initial velocity. The update is the soft-normalized F with tau 0.05, split position steps and three-gain tangent velocity mixing. Retain native-zero branch output projections, zero h-only dynamic matrices, learned scales starting at 0.01, the native attention value embeddings and the ordinary head.

The full run uses the unmodified native optimizer. The static control biases, projection weights and modulation scales share one auxiliary AdamW group with base LR 0.0006 and peak LR 0.000848528137423857 after batch scaling (betas 0.9 and 0.95, zero decay). This explicitly changes the bias LR from the earlier [soft-force research run](https://wandb.ai/yibozhong657-none/nexttransformer-chordres/runs/2574b7dc), whose bias peak was 0.0024494897427831783.

Start from seed 42, step 0, with the original ClimbMix order, 23 blocks, width 1536, FFN 6144, 2,048 context and 1,048,576 input tokens per update. Eight ranks use B8 and eight accumulation steps. Train through all 46,875 updates (49,152,000,000 input tokens). Warm up for 40 updates, then follow the complete native schedule. The evaluation includes all 189 validation points, six full CORE-22 events with 2,048-token left cropping for compatibility with the paired controls, and independent final train and validation BPB. Run the official cumulative audit locally. This is a fork-local experiment selected after earlier research comparisons.

## Verified launch

The [local full-budget W&B run](https://wandb.ai/yibozhong657-none/nexttransformer-chordres/runs/61fdbef5) uses revision `2ad650e7ad71`. The fork's native trainer, optimizer, compute ledger, checkpoint code and CORE evaluator match upstream main at `e4fb048b70a5`. Formal admission returned no findings. All eight ranks verified the same peak LR 0.000848528137 for all 419,738 Chord controls. A fresh optimizer and fresh data cursor started from step zero. The first full validation was 3.16874171 BPB. The first real training update was CE 10.39769365, gradient norm 0.421289, with the auxiliary learning rate 2.12132034e-05 on the 40-step warmup. The target is 46,875 updates with a local cumulative audit after the final evaluations.

## Full-budget endpoint

The local cumulative audit passed with no findings. All 46,875 updates, 49,152,000,000 input tokens, 189 periodic validations, six full CORE-22 evaluations and the independent final BPB evaluation completed. CORE used 2,048-token left cropping. The final eight-rank checkpoint is complete.

| Model | Periodic validation BPB | Independent final validation BPB | CORE-22 (×100) |
| --- | ---: | ---: | ---: |
| Official mHC | 0.661665 | 0.658908 | 35.328 |
| Official Block AttnRes | 0.662963 | 0.660217 | 33.997 |
| Zero-initial-velocity Chord | 0.665465 | 0.662732 | 34.838 |
| Official Dense Baseline | 0.667172 | 0.664460 | 33.095 |
| Official Standard Residual | 0.673959 | 0.671214 | 31.951 |

## Token identity and context at the endpoint

Measured effective controls at step 46,875 on 256 distinct ClimbMix validation sequences (524,288 input tokens). Split the eight data ranks by parity so the folds use disjoint row groups. For each sublayer and coefficient, fit one mean per token ID in one fold and evaluate the other, then reverse the folds. Unseen IDs use the training-fold mean. Pool held-out squared errors across consumed sites and compare against one fitted constant per site. Fixed and structurally unused initial and terminal outputs are omitted.

Token ID predicts 74.5% of the measured variation in lookahead, 72.8% in retention, 61.3% and 49.7% in the parallel and perpendicular velocity writes, and 27.9% and 39.2% in the parallel and perpendicular position steps. The corresponding lookahead scores are 89.7% in the first four layers and 13.6% in the last four. In the last four layers, 69.1% of lookahead variation occurs among occurrences of the same token ID even when each token mean uses the full panel. This analysis describes the trained gates and uses no loss intervention. [Detailed measurements](artifacts/chordres-final-token-type-controls.json)
