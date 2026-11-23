# ChordRes from rest, saving the projected force

This fresh seed-42 run replaces the earlier force-recomputation run. The attention and FFN save their projected BF16 force tensors in GPU memory. The initial velocity is zero and the scalar Chord controls, BF16 residual states and optimizer follow the original run.

The run uses the reference 49.152B-token schedule over 46,875 updates on ClimbMix. Each update consumes 1,048,576 tokens with eight GPUs, per-GPU microbatch 8, eight accumulation steps and context length 2,048. The requested segment ends after update 16,000 (16,777,216,000 tokens). The 8,000- and 16,000-step evaluations and checkpoints are retained. Both use full-context CORE-22. The 16,000-step endpoint can resume on the same schedule.

The calibrated shape and training FLOPs are recorded in the [model configuration](../../configs/residual/chordres_zero_save_force.json) and [cost proof](../../evidence/costs/chordres-zero-save-force-v1/chordres_zero_save_force.json). The code uses the GPU dependencies from the fork's existing uv lock. The run name is `chordres-zero-save-force-49B-seed42` in the [W&B project](https://wandb.ai/yibozhong657-none/nexttransformer-chordres).
