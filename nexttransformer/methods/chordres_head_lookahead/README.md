# Terminal velocity read for ChordRes

The last FFN completes its ordinary Chord write and produces the final BF16 position and tangent velocity. One FP32 scalar rho, initialized at zero, controls a terminal read. Compute h + rho*v in FP32, cast to the ordinary hidden dtype and use the existing final RMSNorm, vocabulary projection and softcap. At rho=0 the read is the original hidden tensor in value. The final FFN retention stays fixed at 0.5 as in the zero-velocity-initialized saved-force baseline. The 46 internal read and write sites are untouched.

The new scalar is in the Chord auxiliary AdamW group. This model keeps the full-attention backbone, alternating attention value embeddings, input smear and explicit zero initial velocity. Its research recipe uses matched random output projections and the elevated control LR.
