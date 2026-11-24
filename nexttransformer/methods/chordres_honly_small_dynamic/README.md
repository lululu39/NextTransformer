# Small h-only corrections to static Chord controls

Each site adds a fixed 0.01 times its projection of the RMS-normalized pre-sublayer position to the trainable static logits. The projection weights start at zero. The two position steps use softplus, velocity retention uses sigmoid, and lookahead and both displacement gains use the signed logits directly. The terminal MLP has three active outputs. The velocity update uses the existing three-gain Chord write.

Use the matched-normal research entry point in `launch/train_chord_control_lr.py`. The static bias group uses the static three-gain rate. The zero-initialized projection stays at the shell's native auxiliary AdamW rate. The scaling is fused into the gate forward and the weight and position derivatives. Bias gradients use the unscaled logit derivative.
