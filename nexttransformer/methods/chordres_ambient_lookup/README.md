# Chord with ambient velocity

Use a spherical position and an unconstrained velocity for every token. The six h-dependent coefficients, split position update, soft force (τ = 0.05), learned modulation, BF16 state, attention value embeddings and smear follow the lookup Chord control. The kernels implement the [nanochat ambient geometry](https://github.com/yibozhong/nanochat/blob/0a1cf8bc8/research/chord-velocity-geometry/REPORT.md) for the width-1536 NextTransformer shell.

The physical lookahead normalizes h + μv using its actual RMS before casting to the state dtype and applying the existing branch RMSNorm. Keep the position on its unit-RMS sphere. Let c = h′ − h and decompose c along the incoming ambient velocity using a squared-RMS floor of 1e−12.

```text
c_parallel = v * mean(v * c) / max(mean(v²), 1e-12)
v′ = β v + γ_parallel c_parallel + γ_perp (c − c_parallel)
```

The default [token lookup](../../../configs/residual/chordres_ambient_lookup.json) uses an independent zero-initialized BF16 embedding in the native token-embedding AdamW group. The [projected-h0 option](../../../configs/residual/chordres_ambient_projected_h0.json) applies a zero-initialized native Muon matrix to the normalized post-smear position. Both feed the ambient vector into the first read. The [zero-start option](../../../configs/residual/chordres_ambient_zero.json) initializes v to exactly zero for every token. All three use the same fused BF16 CUDA read and write with FP32 reductions.

All three 23-layer, width-1,536 presets passed reference cost admission and a bounded eight-H100 sharded forward/backward check on synthetic inputs of length 128. The fused read and write VJPs were checked on non-tangent velocities against autograd. [Geometry checks](../../../tests/test_chordres_ambient.py) · [Official-format cost records](../../../evidence/costs/chordres-ambient-v1)
