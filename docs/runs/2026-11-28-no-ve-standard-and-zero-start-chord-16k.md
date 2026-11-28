# Paired attention VE ablations at 16,000 updates

Queue two independent fresh trainings on the same eight H100s once [the current tangent lookup run](https://wandb.ai/yibozhong657-none/nexttransformer-chordres/runs/3ac13380) finishes its full training and endpoint evaluations. First run [Standard Residual without VE](../../configs/residual/residual_standard_no_ve.json) against [official Standard Residual](../../results/runs/residual_standard.json). Then run [the zero-start tangent Chord without VE](../../configs/residual/chordres_soft_force_zero_no_ve.json) against [the completed VE-on zero-start control](https://wandb.ai/yibozhong657-none/nexttransformer-chordres/runs/61fdbef5).

Both models retain the full-attention 23-layer 1536-wide shell, 12 heads, FFN width 6144, input embedding, learned smear and BF16 recipe. Chord keeps six h-conditioned coefficients, the native auxiliary LR, simple tangent projection, fixed zero initial velocity and soft-force tau 0.05. Each ablation removes 12 attention value tables plus 12 gates (603,981,504 parameters). At initialization the launcher matches the shared tensors to its freshly instantiated VE-on parent on the same device and seed, with native zero attention/MLP output projections. All shared optimizer groups use the parent settings.

| Model | Parameters | Training GFLOPs per token |
|---|---:|---:|
| Standard Residual without VE | 751,829,064 | 4.674149 |
| Chord from rest without VE | 752,248,802 | 4.686898 |

Each model starts at the beginning of the same ClimbMix stream with seed 42, microbatch 8 per rank, accumulation 8, context 2048 and 1,048,576 tokens/update. At step 16,000 each has processed 16,777,216,000 tokens on the original 46,875-update learning-rate and weight-decay schedules. Score the 41,943,040-token periodic validation every 250 steps including step zero, and all questions of all 22 CORE tasks at steps 8,000 and 16,000 with the 2,048-token context cap. At the requested boundary use the native segment-end checkpoint and close W&B cleanly. Finish both CORE evaluations and checkpoint writes on Standard before starting Chord. Both 8k/16k checkpoints are retained. Use the personal W&B project yibozhong657-none/nexttransformer-chordres.

Both ablations passed full-width, full-depth eight-rank sharded forward/backward checks on synthetic data. The paired initialization and optimizer checks and nonzero-branch forward/gradient comparisons also passed. Historical no-VE random-initialization pilot checkpoints remain separate.
