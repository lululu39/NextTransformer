# ChordRes from rest

ChordRes carries a position and a tangent velocity through depth. At the start of each token row, the Standard Residual shell supplies its embedding and smear features. The position is RMS-normalized per token and the initial velocity is exactly zero. The shell's attention value embeddings stay active.

Each attention and FFN reads a lookahead, then writes its projected branch output. With channel mean as the inner product and N as unit-RMS normalization:

```text
q = (h + μ v) / sqrt(1 + mean((μ v)²))
d = N(force) - h
along = mean(d v) / max(mean(v²), 1e-12)
h_next = N(h + α_perp d + (α_parallel - α_perp) along v)
m = β v + (1 - β) (h_next - h)
v_next = m - mean(m h_next) h_next
```

The branch applies native RMSNorm to q. BF16 states cross sublayers; Chord projections, normalization and reductions use FP32. Every site has a scalar μ initialized to zero and two positive softplus steps initialized to 0.05. The first 2L−1 sites learn a sigmoid retention initialized to 0.5. The terminal write uses a constant 0.5 because the LM head reads only h. At 20 layers this adds 159 learned scalars. These parameters belong to the protocol's auxiliary AdamW group.

The full-attention baseline, token smear, value embeddings, Q/K normalization, RoPE, ReLU² FFN, final norm and output head are shared with Standard Residual. The native residual/x0 scales and backout are frozen. Both branch output projections follow the shared zero initialization. The force norm floor is 1e-6 in RMS units, including at initialization.

CUDA training uses a fused BF16 lookahead/read-normalization kernel and a project-only write kernel. Backward reconstructs the force from the final branch projection. The PyTorch audit path replays that projection too. This makes the matrix multiply visible to the training FLOPs ledger. The calibrated full-attention shell is 20 layers at width 1,536 with FFN width 6,272.

Compare against `residual_standard`, `mhc` and `attnres` at the same training updates. The 8,000-update evaluation is an intermediate checkpoint of the original 46,875-update schedule.
