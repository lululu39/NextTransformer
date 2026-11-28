# Zero-start Chord without attention value embeddings

Paired with [soft-force Chord from rest](../chordres_update_soft_force/README.md). Remove all 12 token-indexed attention value tables and their 12 gates. The ordinary input embedding, smear, full causal attention, six h-dependent controls, tangent velocity, soft force and ordinary output head follow the parent. Velocity starts at zero for each token and updates across depth. The paired launcher initializes every shared parameter from a fresh native seed-42 copy of its VE-on control.
