# Standard Residual without attention VE on the matched-normal shell

Start from scratch using [the additive no-VE model](../../configs/residual/residual_standard_no_ve.json) and the matched-normal research launcher. This is the paired Standard Residual for the [no-VE, high-control-LR ChordRes](https://wandb.ai/yibozhong657-none/nexttransformer-chordres/runs/7651aae9). Both use learned input smear, 23 full-attention layers, width 1,536, FFN width 6,144 and depth-scaled random output projections (seed 43). The base seed is 42. They share the remaining embedding, attention, FFN and head initialization and optimizer settings. Standard uses ordinary additions with 751,829,064 parameters; Chord adds 183 control entries.

Use the same native 46,875-step schedule and ClimbMix data order, 2,048-token context, global batch 1,048,576 tokens, eight GPUs, per-rank microbatch eight and eight accumulation steps. Run 8,000 updates (8,388,608,000 tokens) including validation, CORE-22 and checkpointing. No-VE Chord has validation points at 250, 500, 750 and 1,000 and no 8,000-step checkpoint. Report step-matched BPB there. Positive Standard minus Chord BPB means Chord has the lower value.

The [preceding head-lookahead run](https://wandb.ai/yibozhong657-none/nexttransformer-chordres/runs/402de56c) was stopped at logged step 1,830 with checkpoint 1,500 retained. W&B display name: `standard-residual-no-attn-ve-smear-matched-normal-8k-seed42` under yibozhong657-none/nexttransformer-chordres.

## First trained validation

At step 250, the [local no-VE Standard run](https://wandb.ai/yibozhong657-none/nexttransformer-chordres/runs/ff39681c) measured 1.04091092 BPB against 1.03143516 for the matched no-VE ChordRes, a Standard-minus-Chord difference of +0.00947576. Both used 41,943,040 validation tokens at this update. This is the first trained comparison point.
