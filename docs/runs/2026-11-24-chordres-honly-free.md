# Position-only gate with unrestricted Chord controls

Start from the [bounded joint-RMS, native-router-LR control](https://wandb.ai/yibozhong657-none/nexttransformer-chordres/runs/ae5f6592). This run changes two design choices together. Each sublayer generates its controls from RMSNorm(h) alone, using epsilon 1e-6 and a width-D input. All six generated coefficients are direct real values, including signed position steps and velocity retention. The physical lookahead and tangent write continue to use velocity. Initial effective control values are identical to the bounded comparison.

Preserve the 23-layer, width-1,536, FFN-6,144 architecture, random seed-43 attention/MLP output projections, seed 42, zero initial velocity, 12 attention value embeddings, smear, ordinary head and saved BF16 force. The position-only gates have 419,328 projection weights and 273 biases. Use native auxiliary AdamW peak LR 0.000848528137423857, 40-step warmup, eight ranks at B8 and eight accumulation steps, 2,048-token context and the same ClimbMix stream. The target is 8,000 updates on the 46,875-update schedule, with BPB every 250 steps, CORE-22 at 8,000 and the same gradient-stop and persistent reservation policy.

The bounded control was stopped at the requested switch, with its full update-500 checkpoint retained. At that checkpoint its validation BPB was 0.908883.
