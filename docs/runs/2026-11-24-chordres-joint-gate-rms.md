# Joint gate RMSNorm in token-dependent ChordRes

Compare joint RMS normalization of the position and velocity with [velocity-only normalization](https://wandb.ai/yibozhong657-none/nexttransformer-chordres/runs/57313ecf) and the [raw gate](https://wandb.ai/yibozhong657-none/nexttransformer-chordres/runs/81a3c5bc). Both halves share one inverse RMS computed across 2D features with epsilon 1e-6. The gate is parameter-free apart from its existing projection and bias. The lookahead and write consume physical velocity.

Use the same 23-layer, width-1,536, FFN-6,144 shell, attention value embeddings, smear, zero initial velocity and ordinary head. Use saved BF16 forces, random seed-43 output projections, seed 42, and high Chord AdamW peak LR 0.0024494897427831783 with zero control weight decay. Train from scratch on ClimbMix with 2,048 context, eight ranks, B8 per GPU and eight accumulation steps. Stop at 8,000 updates on the original 46,875-update LR schedule, with validation BPB every 250 updates and CORE-22 at 8,000.

[Online training run](https://wandb.ai/yibozhong657-none/nexttransformer-chordres/runs/9fda8c33) uses source revision `3719c72`. The cost audit yielded 4.690242 GFLOPs per input token, 0.382% above the dense anchor. Eight-GPU checks matched all 397 parameter tensors, their trainability and initial logits to the velocity-only model. Fused forward and backward matched the eager reference for zero, small and large velocities.

## First 100 updates

The metrics below use matching optimizer updates. Cross entropy is measured on the training batch.

| Update | Joint gradient norm | Velocity-only gradient norm | Raw gradient norm | Static gradient norm | Joint CE | Static CE |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 10 | 6.196 | 35.48 | 3.91 | 1.639 | 6.99965 | 7.03197 |
| 20 | 0.7557 | 23.66 | 0.4665 | 0.6322 | 5.95822 | 5.98898 |
| 40 | 0.8861 | 3.251e+05 | 9.418 | 0.8177 | 5.51334 | 5.59069 |
| 60 | 0.5774 | 3.863e+09 | 4.691e+10 | 0.3556 | 4.92531 | 4.90236 |
| 80 | 1.175 | 4.503e+11 | 2.158e+17 | 0.215 | 4.44491 | 4.44589 |
| 100 | 2.099 | 7.111e+11 | — | 0.1541 | 4.23309 | 4.12300 |

Joint normalization kept the reported gradient norms within single digits through update 100, including the 40–80 update window in which both token-gate controls developed exploding gradients. Static Chord had lower training CE at update 100. Post-training validation starts at update 250.

## Subsequent instability and shutdown

At updates 120, 130, 170 and 250, production gradient norms rose to 18.1, 842, 6.38e9 and 1.53e13. Validation BPB at 250 was 1.137413, versus 1.011102 in the static control. Cross-token standard deviations at that validation, averaged across the sampled sites, were 3.56 and 3.29 for the two position step sizes and 6.78 and 8.24 for the two displacement-to-velocity gains. The production trainer exited on its nonfinite global gradient norm after the last logged update 390, before the first rolling checkpoint at 500.

The separate diagnostic replay was stopped at the user’s request (last logged update 145). No post-onset joint-RMS weights were captured. On the available two-update smoke checkpoint, stopping gate gradients into velocity changed the initial-position gradient by less than 1% on both probed sequences. Direct post-onset path interventions using the previous velocity-only RMS checkpoint are recorded [with that model](2026-11-24-chordres-gate-v-rmsnorm.md#additional-backward-path-comparisons). Their effect sizes have not been established for the joint-RMS run.
