# ChordRes with a learned token velocity

The saved-force ChordRes model uses an independent vocabulary-by-width table for initial velocity. Each token looks up its velocity, projects it onto the tangent space of the initial unit-RMS hidden state in FP32, and casts the result to the hidden dtype. There is no position table and no velocity normalization. The token module owns both embedding tables so the native optimizer gives them the same AdamW group and BF16 storage.

Velocity weights start from a zero-mean normal with standard deviation 0.02 using a separate generator (weight seed plus two). Shared tensors and the random-projection launcher retain the zero-velocity control's initialization. Chord controls, read/write equations, and saved-force backward follow the control.
