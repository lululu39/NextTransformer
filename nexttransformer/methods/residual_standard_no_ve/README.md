# Standard Residual without attention value embeddings

Run the shared input embedding and learnable smear, then the 23 ordinary full-attention Pre-Norm transformer blocks with additive sublayer updates. Remove the 12 VE lookup tables and their per-head gates. The native residual scaling, x0 reinjection and midpoint backout stay frozen and unused, as in the published Standard Residual. The final norm, LM head and softcap follow the shared shell.

The matched-normal research launcher seeds shared parameters from the VE-on Standard initialization (base seed 42) and redraws the two output projections in each block with seed 43. All retained initial parameters match those in the paired no-VE ChordRes. Standard has no Chord-specific optimizer group.
