# Only the position step reads token-specific branch magnitude

Use a single normalized position step and one complementary velocity update. Each of the 46 sublayers has static μ, scalar position bias b, and amplitude weight w; the 45 consumed velocity outputs each have a static β logit. After the attention or MLP branch returns F, compute α = softplus(b + w log(1 + mean(F²))) per token. Use N(F) for the direction. This adds 46 scalar weights over a constant-step architecture. Zero w recovers its initialization exactly.

Use native zero-initialized attention output and MLP down projections, seed 42, zero v0, μ=0, α=0.05 and β=0.5. All 183 Chord parameters, including w, use the official NextTransformer mHC auxiliary AdamW group (peak 0.000848528137423857, betas 0.9/0.95, zero weight decay). Preserve the 23-layer, width-1,536, FFN-6,144 full-attention shell, 12 VE tables, smear, BF16 saved forces, standard head, 2,048 context, eight ranks at B8, eight accumulation steps and the original 46,875-step schedule. Run one fresh 2,000-update segment and report validation BPB every 250 steps. Keep the existing gradient stop and eight-card handoff.

The previous [h-only single-step run](https://wandb.ai/yibozhong657-none/nexttransformer-chordres/runs/69bbb7bd) used matching zero initialization and native control LR but conditioned all three controls on position. Its last validation was 750. A matched constant-step training run with w fixed at zero would isolate the learned amplitude weight.

[Online training](https://wandb.ai/yibozhong657-none/nexttransformer-chordres/runs/94bbad09) uses source revision `5316ff2`. All eight ranks passed the BF16 fused forward/FP64 derivative oracle and two full-batch updates. The 306 common initialized parameters matched the simplified h-only comparison exactly, as did its starting logits. The model has 1,355,810,751 parameters and the calibrated cost is 4.679439 GFLOPs/token, 0.151% above the shared dense anchor. W&B records the mean absolute amplitude weight and across-token alpha standard deviation on validation.

## Early comparison at 250 updates

Validation BPB on the same 41,943,040 tokens was 1.025440 for the amplitude-only step and 0.997605 for the zero-init h-only three-control model. The older zero-init static two-position-step, one-beta Chord on the same 23-layer shell scored 1.035734. It uses a different position update. Over matched logged updates 110–200, mean training CE was 3.78143, 3.65129 and 3.79298 respectively. The amplitude run’s gradient norm was 0.21650 at update 250.

On the first rank-zero validation batch, the amplitude model’s mean alpha was 0.06780 and its mean within-sublayer, across-token alpha standard deviation was 0.00282, versus 0.04807 in the h-only comparison. The learned amplitude weights had mean and mean absolute value +0.09065. This comparison changes the gate input and removes token-specific lookahead and retention, so it cannot attribute the BPB gap to force amplitude by itself.
