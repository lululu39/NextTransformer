# RMS-normalized gate velocity in token-dependent ChordRes

Compare against the [raw-velocity run](https://wandb.ai/yibozhong657-none/nexttransformer-chordres/runs/81a3c5bc), which recorded a nonfinite gradient norm after update 80. Start from seed 42 with the same random seed-43 output projections. Change only the velocity input to each gate. The normalized input uses one token’s mean squared velocity across its hidden width, epsilon 1e-6, and no learned gain or bias. The lookahead and tangent velocity update retain the physical velocity and its magnitude.

Reuse 23 layers, width 1,536, FFN width 6,144, zero initial velocity, the ordinary LM head, 12 attention value embeddings, smear, saved BF16 forces, and AdamW peak control LR 0.0024494897427831783. All 1,356,649,497 parameters and their initialization match the raw-velocity model. Eight ranks at B8, eight accumulation steps and 2,048-token context yield 1,048,576 input tokens per update. Train 8,000 updates from scratch on the original 46,875-step schedule. Evaluate BPB every 250 steps and CORE-22 at update 8,000.

The [online training run](https://wandb.ai/yibozhong657-none/nexttransformer-chordres/runs/57313ecf) uses source revision `753b013`. The calibration reports 4.690094 GFLOPs per input token, 0.379% above the dense anchor. The eight-GPU preflight reproduced the raw-gate model’s initial logits exactly and matched all 397 parameter tensors and their trainability. The fused gate input and parameter gradients were checked against the eager reference, including zero, small and large velocities.

## First 100 updates

| Update | Training CE | Global gradient norm |
| ---: | ---: | ---: |
| 10 | 7.3906 | 35.48 |
| 20 | 6.1781 | 23.66 |
| 40 | 8.1146 | 3.251e+05 |
| 50 | 8.4445 | 1.83e+06 |
| 60 | 8.3118 | 3.863e+09 |
| 80 | 6.5292 | 4.503e+11 |
| 90 | 5.8635 | 3.597e+12 |
| 100 | 5.6524 | 7.111e+11 |

The model passed the update window in which the raw-velocity run exited. Weightless RMSNorm on the velocity input alone still allowed severe gradient growth under the original control learning rate.
