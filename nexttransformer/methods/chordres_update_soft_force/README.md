# Chord with update-based, soft-normalized force

Treat the attention or MLP branch output as a proposed displacement. For each token, compute its mean square across the model width and use a fixed tau of 0.05.

```text
delta = F / sqrt(mean(F²) + tau²)
h_next = unit_rms(h + alpha_parallel * delta_parallel + alpha_perp * delta_perp)
```

The directional split uses the incoming velocity. The velocity write uses the realized displacement h_next minus h and the existing tangent projection. The position retraction, lookahead read, h-only coefficient generation and learned modulation scales follow the native-zero Chord control. The CUDA backward differentiates the soft force denominator throughout the range. At zero force with equal initial position steps of 0.05 the local force-to-position derivative is the tangent projector at a unit-RMS input.
