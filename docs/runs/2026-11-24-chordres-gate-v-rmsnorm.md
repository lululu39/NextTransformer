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

## Backward-path diagnosis

The last production log was update 160, CE 4.94061, global gradient norm 3.146e+14. Rank zero received SIGINT during backward; peers reported CUDA/NCCL peer-access errors during teardown.

The two-update smoke checkpoint gave a median velocity RMS of 0.0334 (median across the 45 sites with nonconstant input velocity of the per-site token medians). A fresh 42-update replay used the original initialization, token order, eight-rank full batch, optimizer, kernels and schedule. On the same rank-zero microbatch of 16,384 tokens at each tested update, all interventions preserve the forward pass. Only the gradient through the specified gate input or the velocity connection changes. The norm below is with respect to the first normalized hidden state. Each row uses the same weights and tokens across modes. The replay reproduced the instability; individual training losses and gradients can differ from the production trajectory.

| Replay update | Mean CE | Original | Cap gate inverse RMS at 10 in backward | Cap at 1 | Stop gate gradient into velocity | Stop gate gradient into position |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 10 | 7.377268 | 0.00149669 | 0.00142672 | 0.00146152 | 0.00147567 | 0.00158333 |
| 30 | 6.247625 | 0.132008 | 0.00963454 | 0.00407606 | 0.00370999 | 0.0621574 |
| 40 | 8.725633 | 1101.58 | 0.533842 | 0.0349206 | 0.0121647 | 511.732 |

At update 40 the cap of 1 on the inverse-RMS multiplier alone reduced the initial-state gradient by about 31,550×. The physical velocity, forward control values, lookahead and velocity write were identical. This measures local backward sensitivity rather than the effect of training with a modified gate.

## Additional backward-path comparisons

Two distinct fixed 2,048-token ClimbMix training rows were evaluated at the saved **42-update replay checkpoint** using the same-forward gradient interventions. The table reports the L2 norm of the gradient with respect to the first normalized position. Each intervention applies at every sublayer. It modifies only the named Jacobian terms. The losses remain exactly the same within each column.

| Intervention | Row A | Row B |
| --- | ---: | ---: |
| Original | 387.27 | 3097.7 |
| Stop gate gradients into velocity | 0.067652 | 0.059235 |
| Stop gate gradients into position | 29.316 | 114.3 |
| Stop gate-input gradients of both position step sizes | 0.21451 | 0.4489 |
| Clip the two position gains in the state/force VJP at 1 | 259.46 | 448.86 |
| Clip the two displacement-to-velocity gains to [-1,1] in the state/force VJP only | 0.94149 | 1.9922 |
| Freeze both projection directions in the write VJP | 124.01 | 130.86 |
| Cap inverse RMS of the branch force at 1 in backward | 34.529 | 119.91 |
| Cap inverse RMS of the mixed new position at 1 in backward | 57.796 | 1084.5 |

The unmodified losses are 8.45453 and 8.34882. These interventions overlap; their reductions cannot be added as independent shares.
