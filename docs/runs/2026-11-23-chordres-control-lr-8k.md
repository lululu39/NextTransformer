# ChordRes from rest with the nanochat control learning rate

This research run compares against [the completed 8,000-step random-projection ChordRes control](https://wandb.ai/yibozhong657-none/nexttransformer-chordres/runs/1879c1a2). It starts from scratch with seed 42, zero initial velocity, no velocity table, scalar controls and saved BF16 forces. The 23-layer, width-1536, FFN-6144 model has 1,355,810,751 parameters. Both attention output and FFN down projections follow the matched depth-scaled normal initialization with projection seed 43.

The sole optimization change is the peak AdamW rate for the 92 raw step-size entries, 45 retention logits and 46 lookahead coefficients. The rate rises from 0.000848528137423857 to 0.0024494897427831783, the absolute connector rate in the [50M nanochat run](https://wandb.ai/yibozhong657-none/nanochat/runs/38c77ced). The group keeps betas (0.9, 0.95), zero weight decay and the existing 40-step warmup and 46,875-update schedule. The embedding, head, native scalars and Muon matrices retain their respective optimizer groups and rates.

Use the [research entry point](../../launch/train_chord_control_lr.py) and the [existing saved-force ChordRes config](../../configs/residual/chordres_zero_save_force.json). The training segment is 8,000 updates of 1,048,576 ClimbMix tokens with 2,048 context, eight data-parallel ranks, per-rank microbatch eight and eight gradient accumulations. Record the full scheduled validation BPB and the 8,000-step CORE-22 and checkpoint. The W&B name is `chordres-v0zero-scalar-lr0.00245-matched-normal-8k-seed42`.

Training stopped at update 4,000 (validation BPB 0.79343645, complete checkpoint) for the [no-attention-VE, smear-retained comparison](2026-11-23-chordres-no-attention-ve-8k.md).
