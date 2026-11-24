# Position step from the full branch output

Each attention and MLP sublayer owns one width-D projection vector w and one position bias b. Given the raw branch vector F at a token:

```text
alpha = softplus(b + dot(w, F))
h_next = N(h + alpha * (N(F) - h))
v_next = P_at(h_next, beta * v + (1-beta) * (h_next - h))
```

Only alpha varies across tokens. Mu and beta are static per site; the last MLP fixes its unused beta to 0.5. The projection reads the BF16 branch output before RMS normalization, so two vectors with identical RMS can produce different step sizes. The normalized force supplies the update direction. Native zero output projections and w=0 initialize the model at the constant step of 0.05. The fused Triton write computes the dot product alongside force normalization. Its VJP includes the force-gradient contribution from the gate and a tiled full-width projection-weight reduction.
