# Force-amplitude-conditioned position step

Each attention or MLP site has a static signed lookahead μ, a static sigmoid velocity-retention β, and a single position step. Only the step depends on the token. Its two parameters are a bias and a scalar weight:

```text
r² = mean(F²)                 # across hidden channels of one token
α = softplus(b + w log(1 + r²))
h_next = N(h + α (N(F) − h))
v_next = project_at(h_next, β v + (1−β)(h_next−h))
```

Use b = log(expm1(0.05)), w = 0, μ = 0 and β = 0.5 initially. Every site starts as a constant-step Chord. There are 46 amplitude weights for the 46 sublayers and 45 learned velocity retentions; the final MLP reads out h. The fused write computes both the force normalization and the amplitude feature from a shared FP32 reduction. Its backward differentiates through the normalized direction and the amplitude input to α. Force and carried states remain BF16.
