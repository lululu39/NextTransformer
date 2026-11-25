# Initial tangent velocity from the Chord position

After the normal embedding, smear and unit-RMS position initialization, apply a single shared, bias-free width-by-width matrix to the stored h0. Remove the component parallel to h0 using FP32 reductions and store the resulting velocity in the hidden-state dtype. Its magnitude remains learnable.

```text
u = W_v h0
v0 = u - h0 * mean(h0 * u) / mean(h0²)
```

The matrix starts at exactly zero and follows the ordinary two-dimensional Muon parameter group. Every subsequent gate, position write and velocity update follows the native-zero update-based soft-force Chord recipe (tau 0.05).
