# No-VE mHC on the matched-normal comparison

Run a fresh nanochat-style four-stream mHC on ClimbMix. The [no-VE Standard Residual](https://wandb.ai/yibozhong657-none/nexttransformer-chordres/runs/ff39681c) stopped after validation and a complete checkpoint at step 1,000. At that step Standard scored 0.869074 BPB and [no-VE ChordRes](https://wandb.ai/yibozhong657-none/nexttransformer-chordres/runs/7651aae9) scored 0.860166 BPB.

The [mHC model](../../nexttransformer/methods/mhc_nanochat_no_ve/README.md) has 23 layers, width 1,536, FFN width 6,016 and 749,593,895 parameters. Attention has no VE tables or gates. The native input smear is retained. Router read biases start at -log(3), write bias at zero, mixing bias at identity, and dynamic scale at 0.01. A learned sigmoid readout mixes the four terminal streams. All routers and the final readout use AdamW, peak LR 0.0008485281, betas 0.9/0.95 and zero weight decay.

Use seed 42 for the backbone, seed 43 for the depth-scaled random attention output and FFN down projections, eight GPUs, per-GPU microbatch eight, eight accumulation steps, context 2,048 and 1,048,576 tokens per optimizer step. Preserve the 46,875-step native LR schedule. The requested segment runs to 8,000 updates (8,388,608,000 tokens) including validation, CORE-22 and checkpoint. Its audited cost is 4.674470 GFLOPs per token (+0.044% versus the NextTransformer anchor). Research results are compared locally; the no-VE Standard and Chord validation histories end at 1,000. The W&B name is `mhc-nanochat-no-attn-ve-smear-matched-normal-8k-seed42` in the [research project](https://wandb.ai/yibozhong657-none/nexttransformer-chordres).

## Startup measurements

[Online run](https://wandb.ai/yibozhong657-none/nexttransformer-chordres/runs/f1a480a8). The eight-rank preflight completed two real optimizer updates. The auxiliary group contained the routers and final readout with peak LR 0.000848528137423857, zero weight decay and finite nonzero readout gradients. The initial backbone matched a same-shaped Standard reference. The training pass started freshly and measured 3.16854589 BPB before the first update. Training CE at step 10 was 7.181298. Validation events log `mhc_minus_chord` and `mhc_minus_standard` across common trained points, including 250, 500, 750 and 1,000.

The nanochat-style mHC training was stopped at logged step 140. It was superseded by the [official mHC ablation](2026-11-24-official-mhc-no-ve.md); its metrics are kept under their original run ID.
