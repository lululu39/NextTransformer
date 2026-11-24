# Bounded token-dependent controls for joint-RMS ChordRes

Each token and sublayer retains the [joint position/velocity gate normalization](../chordres_token_controls_joint_rms/). For the six gate logits including their trainable biases:

```text
mu = 3 * tanh(logit_mu / 3)
alpha_perp, alpha_parallel = sigmoid(logits_alpha + logit(initial_alpha))
beta = sigmoid(logit_beta)
gamma_parallel, gamma_perp = 2 * sigmoid(logits_gamma)
```

The zero-start projection and raw biases of 0 (mu, centered alpha and beta) and -log(3) (gamma) yield effective initial controls of 0, 0.05, 0.05, 0.5, 0.5 and 0.5. The terminal MLP retains fixed velocity coefficients of 0.5. The nonlinearities and their derivatives run inside the Triton read kernels. The physical velocity, lookahead read, write and tiled parameter-gradient reduction use the same equations as the joint-RMS control.

The position sigmoid uses centered odds with the same FP32 initial step as the control (0.049999985843896866). This preserves the initial model through BF16 rounding across depth.
