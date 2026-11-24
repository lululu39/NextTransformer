# Bounded Chord controls with the native mHC router learning rate

Compare the [bounded Chord high-LR run](https://wandb.ai/yibozhong657-none/nexttransformer-chordres/runs/f0d32bb6) with a fresh run using the native NextTransformer auxiliary AdamW group. The joint h/v projection weights and trainable control biases use the same peak LR as the official mHC router: 0.0006 × sqrt(1048576 / 524288) = 0.000848528137423857, down from 0.0024494897427831783. The learning-rate schedule has a 40-update warmup. The group retains betas (0.9, 0.95), epsilon 1e-10 and zero weight decay.

Use the same [bounded model configuration](../../configs/residual/chordres_joint_gate_bounded.json), input data order, random seed 42 and seed-43 output projections, BF16 saved-force kernels, B8 per rank, eight GPUs and eight accumulation steps. The target is 8,000 updates on the original 46,875-update schedule, validation BPB every 250 steps and CORE-22 at 8,000. Inherit the two-consecutive-10,000 gradient stop and persistent GPU leases. The [research launcher](../../launch/train_chord_native_lr.py) takes the control LR directly from the native group.

The previous high-LR run stopped after logged update 650 and retained a full checkpoint at 500. At update 500 its training CE was 3.00116 and validation BPB was 0.917362. The static Chord control recorded 2.98919 and 0.914121.

[Online training](https://wandb.ai/yibozhong657-none/nexttransformer-chordres/runs/ae5f6592) uses source revision `a6ce2cb`. Eight-rank preflight verified the native control peak on every rank and observed learning rates of 2.12132034356e-5 and 4.24264068712e-5 at the first two updates. All 397 initial parameter tensors, their trainability and the checked initial logits matched the high-LR control exactly.

## Why compare the router learning rate

Mean differences in logged training CE (bounded high LR minus static Chord, same updates and data) were −0.10052 at updates 110–200, −0.00085 at 310–400 and +0.02103 at 610–650. Both CE curves decreased over these windows. The former run’s largest global gradient norm from update 250 to its stop was 3.58. At update 500 its BPB was 0.917362, compared with 0.914121 for static Chord and 0.909574 for the official NextTransformer mHC. The last comparison changes architecture as well as optimization.

A frozen forward-only check of high-LR bounded checkpoint 500 on 131,072 ClimbMix validation tokens measured how often each token-level control fell within 1% of either endpoint of its own allowed interval. The statistics cover all consumed sites (46 position/read sites and 45 velocity writes) and eight validation streams.

| Control | Near lower bound | Near upper bound |
| --- | ---: | ---: |
| mu | 14.02% | 10.27% |
| alpha_perp | 5.27% | 0.00% |
| alpha_parallel | 39.56% | 7.46% |
| beta | 12.11% | 27.86% |
| gamma_parallel | 37.56% | 33.48% |
| gamma_perp | 11.35% | 11.93% |

Fixed-checkpoint mean CE was 3.234862. With all other weights and the raw biases fixed, scaling the input-dependent projection by 0.9 increased CE by 0.020891. Scaling it by 1.1 increased CE by 0.009003. Scaling only the position-step projection by 0.9 increased CE by 0.020123, compared with 0.000127 for the displacement-to-velocity projection. This records sensitivity of the trained checkpoint. Early saturation as a cause of slower subsequent learning remains a hypothesis.

## Matched checkpoints through 500 updates

| Update | Native router LR BPB | High router LR BPB | Static Chord BPB | Native router LR training CE | High router LR training CE |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 250 | 0.990882 | 1.001307 | 1.011102 | 3.259627 | 3.299800 |
| 500 | 0.908883 | 0.917362 | 0.914121 | 2.969891 | 3.001158 |

The table below aggregates training CE on the paired, logged updates (every ten steps). It uses the same data, seed and model in the two bounded runs.

| Updates | Native LR CE | High LR CE | Static CE | Native minus high |
| --- | ---: | ---: | ---: | ---: |
| 10–100 | 5.127891 | 5.119966 | 5.197173 | +0.007925 |
| 110–200 | 3.559141 | 3.586780 | 3.687305 | -0.027639 |
| 210–300 | 3.246779 | 3.282900 | 3.314798 | -0.036121 |
| 310–400 | 3.112119 | 3.150604 | 3.151458 | -0.038485 |
| 410–500 | 3.029279 | 3.063015 | 3.056262 | -0.033736 |

Fraction of token/site control values within the outer 1% of either end of their allowed interval, measured on the same 131,072 validation tokens at update 500:

| Control | Native LR | High LR |
| --- | ---: | ---: |
| mu | 5.13% | 24.29% |
| alpha_perp | 0.38% | 5.27% |
| alpha_parallel | 46.35% | 47.02% |
| beta | 25.85% | 39.97% |
| gamma_parallel | 38.87% | 71.04% |
| gamma_perp | 10.88% | 23.27% |

[Matched gate statistics and same-token CE perturbations](artifacts/chordres-bounded-router-lr-step500.json). Results are from one paired seed.

## Sensitivity to the gamma ceiling

On the same 131,072 validation tokens, identify the token/site writes with an effective gain above 1.98 on the original forward pass. With masks fixed to those tokens, add or subtract 0.10 from that coefficient alone in the physical write and recompute downstream states. Each column reports the change in mean CE.

| Router LR | Original CE | Parallel +0.10 | Parallel −0.10 | Perpendicular +0.10 | Perpendicular −0.10 |
| --- | ---: | ---: | ---: | ---: | ---: |
| high | 3.234862 | -0.000017 | +0.000017 | +0.000017 | +0.000003 |
| native | 3.204241 | +0.000077 | +0.000043 | +0.000022 | +0.000085 |
