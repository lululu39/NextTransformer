# ChordRes with an unsplit step and a single velocity gate

Each sublayer reads normalized position to produce a signed preview coefficient μ, a positive softplus position step α and a sigmoid retention β. Let N denote unit-RMS normalization and P the projection onto the new tangent plane.

```text
q = N(h + μv)
h_next = N(h + α (N(force) - h))
v_next = P_at(h_next, βv + (1−β)(h_next−h))
```

The full displacement uses one step size; the velocity receives the full realized displacement with weight 1−β. The last MLP learns two outputs for the position and preview, with fixed unused retention 0.5. At width 1,536 there are 137 scalar biases and 210,432 position-gate weights. Zero initial gate weights match the original scalar biases μ=0, α=0.05 and β=0.5. The fused read and write kernels use the same BF16 state boundaries as the other Chord variants.
