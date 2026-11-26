# Token-indexed initial tangent velocity

Each token looks up one row of an independent 32,768 by 1,536 velocity table. Project this vector onto the tangent plane of the normalized post-smear initial position, using the same Triton projection and BF16 output as the projected-h0 control. The zero-initialized table uses the native token-embedding AdamW group. All subsequent reads and writes follow the same soft-force Chord equations.
