# Fresh h-only three-control Chord through 2,000 updates

Repeat the [native-zero single-step h-only model](https://wandb.ai/yibozhong657-none/nexttransformer-chordres/runs/69bbb7bd) from update zero with its original scientific revision `98321666f4b56e39447b3ee0d8b59e935c310df4`, identical model config, seed 42, ClimbMix order, batch and 46,875-update learning-rate schedule. Each token at each sublayer reads RMSNorm(h) to predict a signed lookahead mu, a single softplus position step alpha and a sigmoid velocity retention beta; the terminal MLP has two active outputs. The native auxiliary AdamW group uses peak LR 0.000848528137423857. The branch output projections and token-gate matrices start at zero. The standard force normalization, unit-RMS position and complementary velocity write remain in place.

The previous h-only run stopped on relative BPB at update 750. This fresh run uses a fixed 2,000-update segment, validation BPB every 250 and the existing pre-update gradient failure guard. Its W&B validation curves compare paired updates to the earlier h-only run, the two force-gate runs, static Chord and the official residual baselines. The full-force run was stopped following its 250-step validation. That comparison also differs in whether mu and beta are token dependent.

[New online run](https://wandb.ai/yibozhong657-none/nexttransformer-chordres/runs/345e22ad). Eight-rank preflight passed the gated read/write oracle and two real optimizer updates. Runtime provenance matched the earlier h-only model, protocol, optimizer, data and tokenizer hashes, precision and global batch, with a fresh optimizer and stream.

## Reproducibility audit at update 500

The original and repeat used identical source revision, model and protocol hashes, training configuration, Torch/CUDA stack, tokenizer and data fingerprints, host, batch shape and initial validation BPB. The repeat uses a fixed 2,000-update endpoint and different comparison logging. The rank-local stream identity, cursor, pending document buffers, consumed row count, routing state and saved Python, NumPy, Torch CPU and CUDA RNG states matched exactly on all eight ranks at preflight update 2 and training update 500.

The first training CE (10.397693648934364) and full gradient norm (5536.83544921875) matched exactly. The first recorded gradient norm difference occurred on update 2 (original 62.962093353271484, repeat 62.96207046508789). At 250 and 500, repeat BPBs were 0.997101874 and 0.909024094, versus 0.997605286 and 0.907599081 originally.

Both runs selected FA3. In an isolated BF16 causal FA3 test, each of eight H100s repeated the forward/backward call three times with identical Q, K, V and upstream gradient (two 2,048-token sequences, 12 heads of dimension 128). Every forward output matched bitwise. Each repeat differed from the first in 43–101 components of dQ out of 6,291,456 per device, with maximum absolute difference 0.00048828125 across devices and repeats. The sampled dK and dV matched. This demonstrates a source of run-to-run rounding differences; the whole-network drift was not isolated to one operator.

[Velocity and displacement alignment](2026-11-24-honly-ve-velocity-alignment.md) measures the paired VE effect at updates 500 and 1,000.

[VE attention projection-only test](2026-11-24-honly-ve-projection-only.md) compares retaining old velocity to selectively omitting the VE displacement.
