# ChordRes from rest without attention value embeddings

Use the same 23-layer saved-force ChordRes as the VE-on control. Remove the alternate-layer token-value tables and their per-head VE gates. Every attention layer now takes V from the usual projection of its hidden input. The input token embedding and learned input smear remain active. The method inherits the scalar Chord updates, zero initial velocity, full attention and saved BF16 forces.

For paired research training, the matched-normal launcher seeds common tensors from the VE-on reference initialization before applying the depth-scaled random output projections. This matches the shared backbone, controls and RNG state despite the change in module allocation. Only the no-VE model participates in training. Its checkpoint and optimizer contain no VE tables or VE gates.
