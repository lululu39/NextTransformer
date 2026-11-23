# Token initial velocity on the matched-normal ChordRes control

Train from step zero to 8,000 updates (8,388,608,000 tokens) with the [23-layer token-velocity model](../../configs/residual/chordres_vembed_save_force.json). The comparison is the completed [random-projection, zero-velocity ChordRes run](https://wandb.ai/yibozhong657-none/nexttransformer-chordres/runs/1879c1a2).

Use seed 42 and the same ClimbMix data order, 2,048 context, per-GPU microbatch 8, eight accumulation steps and eight GPUs. The 46,875-step LR schedule, BF16 states, scalar controls, saved branch forces and random output projections (seed 43) match the control. The only new parameter is the untied 32,768 by 1,536 token velocity table. It uses seed 44, normal std 0.02, and the same BF16 storage and AdamW group as token embeddings (peak LR 0.30, betas 0.8/0.995, decay 0.001). Project onto the tangent space before storing the BF16 state. There is no position embedding for velocity.

The run name is `chordres-token-vembed-save-force-matched-normal-8k-seed42`. Use the research entry point `python -m launch.train_matched_normal` and retain the step-8,000 checkpoint, validation BPB and CORE-22. The table adds 50,331,648 parameters; its contribution combines token-conditioned initial velocity with extra vocabulary capacity.

The run was stopped at the first complete 500-step checkpoint. See the [validation ablations and layer diagnostics](2026-11-23-chordres-early-dynamics.md).
