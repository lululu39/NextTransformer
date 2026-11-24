# Token-dependent ChordRes, 8,000 updates

The paired static model is the [three-gain ChordRes run](https://wandb.ai/yibozhong657-none/nexttransformer-chordres/runs/a710a057). It stopped after writing its 2,500- and 3,000-update checkpoints. Compare validation BPB every 250 updates through the common 3,000-step window.

Train from scratch using the [token controls](../../nexttransformer/methods/chordres_token_controls/) on the fixed 23-layer, width-1,536, FFN-6,144 shell. The 838,656 new weights use the existing Chord AdamW group at peak LR 0.0024494897427831783, with zero weight decay. The static biases and all shared weights retain their original initialization. Use seed 42 and seed-43 normal attention and MLP output projections, zero initial velocity, 12 attention value embeddings, input smear, saved BF16 forces and the ordinary LM head.

Train on the original ClimbMix stream at 2,048 context, B8 per rank, eight ranks and eight accumulation steps. Stop after 8,000 updates, or 8,388,608,000 tokens, of the original 46,875-update schedule. Evaluate validation BPB every 250 steps and CORE-22 at 8,000. Record the dispersion of each learned control on the first validation batch at rank zero. The comparison uses the same W&B entity and project as the static run.

The eight-GPU preflight matched all 351 shared parameter tensors including their trainability and reproduced the static model’s initial logits exactly on the check sequence. Both input halves of the new projection had nonzero gradients on each of the two full-batch training updates. The initial training CE matched the static control (10.39548248).

[Training run](https://wandb.ai/yibozhong657-none/nexttransformer-chordres/runs/81a3c5bc) uses source revision `3fc67c45e24c40f7b1d4827237e03d34268ce5ce`. The static control reached 0.798980 validation BPB at update 3,000.

## Gradient failure

The original run stopped on its nonfinite total gradient norm after its last logged update, 80, before the first scheduled post-training validation at 250. At updates 40, 50, 60 and 80 the norms were 9.42, 4.72e7, 4.69e10 and 2.16e17. The static control recorded 0.818, 0.394, 0.356 and 0.215 at the same updates.

A bounded eight-GPU replay used the original seed, data order, full batch, optimizer, initialization, precision and kernels. The exact minibatch trajectory varied slightly. On the first rank-zero microbatch at update 44, the maximum per-token velocity RMS over the 46 sublayers exceeded 10,000. At update 50 it reached 650,631, and the largest absolute token-specific velocity gains exceeded one million. The first attention site's gate weight received the largest full-batch parameter gradient at that update.

Each diagnostic rank evaluated the same 8 × 2,048 training tokens and the same update-50 weights. The measured quantity below is the L2 norm of the gradient of mean CE with respect to the initial normalized hidden state. Backward interventions preserve the forward values at every sublayer. Forward interventions restore the selected gate contributions to their learned bias or remove a selected input, then recompute all subsequent sublayers.

| Same-batch intervention | Mean CE | Initial hidden gradient norm |
| --- | ---: | ---: |
| Original backward | 6.180725 | 2,244.18 |
| Remove only the gate derivatives with respect to input velocity | 6.180725 | 0.046984 |
| Remove only the gate derivatives with respect to input position | 6.180725 | 655.98 |
| Remove only the position-gain derivatives from the gate input path | 6.180725 | 1.98591 |
| Original forward control for the forward interventions | 6.180725 | 2,248.43 |
| Remove the velocity input from gate predictions, retaining position conditioning | 5.954073 | 0.006058 |
| Use only the learned biases for the two displacement-to-velocity gains | 5.987340 | 0.005588 |

This localizes the large gradient on this batch to the feedback from unnormalized velocity into the controls. The position stream has unit RMS, while the state update constrains velocity to the tangent plane but leaves its magnitude free. The unbounded displacement-to-velocity gains can amplify the velocity subsequently read by the next gate. The learning-rate warmup ends at update 40. Its independent contribution has not been measured.

The replay reached the finite-gradient check at update 66. Every individual parameter gradient was finite, with maximum absolute entry 7.22e19. Accumulating the global L2 norm in FP64 gave 2.39e21. FP32 norm accumulation overflowed on these already large gradients. The production run did not record per-parameter statistics at its failing update.
