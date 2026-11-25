# Chord with learned modulation scales

Each Chord site reads RMS-normalized h. The effective logits are the static biases plus a learned scale times the site-specific token projection. Lookahead has its own scale, the two position steps share one, and the three velocity coefficients share one. The final MLP needs only the lookahead and position scales. Apply the same scalar activations as in static three-gain Chord.

All gate matrices start at zero. Each scale starts at 0.01. Use the native-zero research entry point in `launch/train_chord_zero_init.py`. Attention outputs and MLP down projections use the native zero initialization. Static biases use the high static Chord AdamW rate. The projection weights and their scales share the native auxiliary rate. The fused read differentiates both factors without dividing by the scale.
