# Direct token VE impulse

At each attention site with a value embedding, the ordinary static Chord position update produces the next unit-RMS position. The velocity uses the same token's raw lookup from that layer's existing value table.

```text
e = VE_layer[token]
v_next = project_tangent(h_next, beta * v + (1 - beta) * (h_next - h) + lambda * e)
```

All sites share one signed scalar lambda initialized at zero. The attention branch retains its ordinary VE gate. Attention sites without a value table and all FFN sites use the usual saved-force Chord write. The Triton implementation fuses the additional term and its gradients into the affected writes.
