# ChordRes with RMS-normalized velocity for token controls

Each site supplies the gate with `v / sqrt(mean(v²) + 1e-6)`, where the mean is over one token’s hidden width. The normalization has no learned gain or bias. The zero initial velocity produces a zero gate input. Position conditioning, all control parameterizations, and the high control LR follow the [raw-velocity experiment](../chordres_token_controls/). The lookahead and tangent update use physical velocity at its original magnitude.

The BF16-state Triton read normalizes the gate input in FP32. Its backward applies the RMSNorm Jacobian to the gate contribution to the velocity gradient and adds the physical lookahead gradient. The shared velocity weight gradient uses per-token inverse RMS inside the existing tiled reduction, without saving a second normalized velocity tensor. The write uses the original three-gain Triton kernel.
