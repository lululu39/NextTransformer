# Direct VE velocity write

For the 12 attention blocks with a native value table, reuse the raw embedding for the token at that position. Add it at full strength before projecting the new velocity onto the tangent plane. Attention continues to consume the same lookup with its native gate. The embedding table receives gradients from both paths.

```text
e = VE_layer[token]
v_next = project_tangent(h_next, beta * v + (1 - beta) * (h_next - h) + e)
```

The impulse introduces zero parameters. The Triton write fuses the addition, projection and their derivatives. All FFN writes and the remaining attention writes use the standard saved-force Chord implementation.
