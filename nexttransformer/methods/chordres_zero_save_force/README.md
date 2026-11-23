# ChordRes from rest with saved forces

This method uses the scalar zero-initial-velocity ChordRes equations, initialization and fused BF16 kernels from [ChordRes from rest](../chordres_zero/README.md). The attention and FFN execute their ordinary output projections. Each Chord write retains the projected BF16 force on the GPU so backward can read it directly. The PyTorch reference and the training FLOPs audit use the same single projection per sublayer.
