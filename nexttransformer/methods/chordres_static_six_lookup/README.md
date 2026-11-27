# Static six-control tangent Chord

The paired dynamic model adds a learned h-dependent correction to a shared set of control logits. This ablation retains the static logits and their initial values and removes the projection rows and modulation scales. Each of the 45 sublayers that write a consumed velocity learns six scalars. The final MLP learns its three read and position scalars, giving 273 trainable control parameters in total. All use the same auxiliary AdamW settings as the dynamic control's static biases.

The inherited forward path uses token-indexed initial tangent velocity, BF16 states, a physical lookahead read and split spherical position updates driven by the soft force at τ=0.05. The velocity update retains the same tangent projections and independent parallel/perpendicular writes. CUDA uses the existing fused scalar read and soft-force write. [Matched dynamic model](../chordres_soft_force_token_velocity/README.md)
