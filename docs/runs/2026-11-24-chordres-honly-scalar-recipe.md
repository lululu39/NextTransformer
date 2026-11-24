# Position-only Chord controls with the original scalar constraints

Start from scratch on the 23-layer Chord shell. Read only normalized position into the token gate and use the original [static three-gain Chord](../../nexttransformer/methods/chordres_velocity_gains/) coefficient activations. Mu and both gamma values are direct, both alpha values use softplus, and beta uses sigmoid. The original scalar biases and a zero initial position projection reproduce the scalar model at initialization. Physical velocity keeps its scale for the lookahead and tangent update.

The superseded [all-affine interpretation](https://wandb.ai/yibozhong657-none/nexttransformer-chordres/runs/9e441e9a) stopped after its logged update 50. Its metrics belong to that historical parameterization. Compare against the [bounded joint h/v run at the same LR](https://wandb.ai/yibozhong657-none/nexttransformer-chordres/runs/ae5f6592) and the [static Chord control](https://wandb.ai/yibozhong657-none/nexttransformer-chordres/runs/a710a057).

The native auxiliary AdamW peak is 0.000848528137423857. Keep seed 42, seed-43 normal output projections, 12 attention value embeddings, smear, zero initial velocity, the ordinary head and saved BF16 forces. Use eight GPUs, B8 per GPU with eight accumulation steps, 2,048 context, ClimbMix data and 8,000 updates on the 46,875-update schedule. Preserve validation every 250 updates, CORE-22 at 8,000, pre-update gradient failure capture and persistent memory reservations.

[Online training](https://wandb.ai/yibozhong657-none/nexttransformer-chordres/runs/e266c628) uses revision `bb461be`. All 351 shared parameter tensors and initial logits matched the static scalar control exactly. The first two full-batch updates passed on all eight ranks. The position-only gates add 419,328 projection weights to the static scalar model, yielding 1,356,230,169 total parameters and 4.687258 GFLOPs per input token (0.318% above the dense anchor).

## Comparison at 1,000 updates

Validation BPB on 41,943,040 fixed tokens. Lower values are better. Published reference curves use the same evaluation protocol. The bounded Chord at native LR was stopped before update 1,000.

| Model | 250 | 500 | 750 | 1,000 |
| --- | ---: | ---: | ---: | ---: |
| H-only Chord with original scalar constraints | 0.986739 | 0.907240 | 0.868241 | 0.847444 |
| Bounded joint Chord at native LR | 0.990882 | 0.908883 | 0.869800 | — |
| Static three-gain Chord | 1.011102 | 0.914121 | 0.870654 | 0.848149 |
| [Official mHC](../../results/runs/mhc.json) | 1.014167 | 0.909574 | 0.868252 | 0.846212 |
| [Official Block AttnRes](../../results/runs/attnres.json) | 0.984686 | 0.901601 | 0.864181 | 0.843815 |
| [Official Standard Residual](../../results/runs/residual_standard.json) | 1.007384 | 0.916649 | 0.878293 | 0.863117 |

Mean training CE across logged updates 910–1,000 was 2.804573 for h-only Chord and 2.806023 for static Chord. Across all 100 individual updates from 901 to 1,000 the global gradient norm had median 0.90088, maximum 1.82161, and value 0.60814 at 1,000. The model and distributed state checkpoint at 1,000 is complete.

## Branch-output magnitude at 1,000 updates

On eight validation streams (131,072 tokens total), record the RMS of the raw attention or MLP output at each token before the Chord write normalizes its target. P90/P10 is computed over tokens within each sublayer. Correlations are computed over those same tokens separately per sublayer. The table reports medians across sublayers in the named group.

| Sites | P90/P10 branch RMS | Absolute correlation of log branch RMS with alpha perpendicular | With alpha parallel | With realized step RMS |
| --- | ---: | ---: | ---: | ---: |
| attention | 1.346 | 0.082 | 0.101 | 0.108 |
| mlp | 2.088 | 0.260 | 0.136 | 0.249 |
| all | 1.698 | 0.170 | 0.123 | 0.176 |

On the first 32,768 validation tokens, multiplying the output of every attention and MLP by 8 at the Chord write gave bitwise identical final hidden states on all eight GPUs. The measurement establishes scale invariance in these examples. It does not measure the accuracy effect of training a step-size gate with access to branch magnitude.
