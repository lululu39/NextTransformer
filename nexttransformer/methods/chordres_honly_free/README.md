# Position-only affine Chord controls

Each token and sublayer uses a zero-initialized projection of normalized position. For an input of width D, the weight has shape [6, D] at ordinary sites and [3, D] at the terminal MLP.

```text
controls = bias + W RMSNorm(h)
```

RMSNorm uses epsilon 1e-6. All six consumed coefficients are direct signed real outputs. Their initial values are 0 for lookahead, 0.05 for both position steps and 0.5 for old-velocity retention and both displacement gains. The terminal MLP has three position/read controls and fixed unused velocity coefficients. The BF16 lookahead still reads physical velocity, and the original tangent write updates it at every sublayer. Triton fuses the projection, lookahead and their VJPs, including a single-stream tiled parameter-gradient reduction.
