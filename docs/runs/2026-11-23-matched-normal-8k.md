# Matched random projection initialization at 8,000 updates

The initial comparison scheduled ChordRes from rest with saved BF16 forces first, then Standard Residual. Each model starts from scratch and stops after its own step-8,000 validation, full CORE-22 evaluation and checkpoint. These are research runs using the native 46,875-step learning-rate schedule. Each model consumes 8,388,608,000 ClimbMix tokens in this segment.

Both models have 23 full-attention layers, width 1,536, FFN width 6,144 and 2,048-token context. Use the existing GPU uv stack, BF16 states, eight GPUs, eight sequences per GPU, eight accumulation steps, and 1,048,576 tokens per update. ChordRes uses zero initial velocity, scalar controls, the existing force RMS floor of 1e-6, and no force recomputation.

The research launcher uses the common seed-42 backbone. In layer order it redraws just the attention output and FFN down projection weights from a zero-mean normal with standard deviation 0.02/sqrt(2*23). Both runs use the same dedicated seed-43 generator for these draws. This preserves the other initialized tensors and the original training RNG stream. The launcher records the initialization in checkpoint and W&B metadata.

Use `python -m launch.train_matched_normal` with `--research` and either the [ChordRes configuration](../../configs/residual/chordres_zero_save_force.json) or the [Standard Residual configuration](../../configs/residual/residual_standard.json). Runs are grouped under `matched-normal-8k` in the [W&B project](https://wandb.ai/yibozhong657-none/nexttransformer-chordres).

## Revised training order

After the 1,000-update initialization comparison, the random-initialized ChordRes run continues through its step-8,000 validation, CORE-22 and checkpoint. The random-initialized Standard Residual run is deferred. Its earlier eight-rank preflight was a single test update.

Next, start a separate ChordRes run from step zero with the native zero-initialized attention output and FFN down projections. Use the [23-layer saved-force configuration](../../configs/residual/chordres_zero_save_force.json) with the native trainer. Initialize model, optimizer, schedule and data stream afresh at seed 42. This run also ends at update 8,000 on the original 46,875-update LR schedule, after validation, CORE-22 and checkpointing. The W&B display name is `chordres-zero-init-save-force-8k-fresh-seed42`. The published Standard Residual curve is available in the same project.
