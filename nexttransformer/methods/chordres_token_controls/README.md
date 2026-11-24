# Token-dependent ChordRes

Each attention or MLP site reads its input position `h` and tangent velocity `v` and predicts six coefficients per token. Tokens share the site's weights. The read and the write use the same coefficients.

```text
raw = bias + W_h h + W_v v
mu = raw[0]
alpha_perp, alpha_parallel = softplus(raw[1:3])
beta = sigmoid(raw[3])
gamma_parallel, gamma_perp = raw[4:6]
```

The biases retain the [three-gain ChordRes](../chordres_velocity_gains/) initialization. Both weight matrices start at zero. At the final MLP only the first three coefficients affect the ordinary LM head. This site predicts those three and uses fixed velocity coefficients of 0.5. At width 1,536 this adds 838,656 weights across the 46 sites.

The forward read kernel fuses the two input projections, biases and nonlinearities with lookahead and RMSNorm, and emits six FP32 values per token for the write. The write kernel consumes those values directly. The read backward combines the write and lookahead gradients and computes state gradients. A two-stage Triton reduction computes the shared weight and bias gradients with FP32 accumulation. Residual state and saved force use BF16. The eager reference supports calibration and gradient comparisons.
