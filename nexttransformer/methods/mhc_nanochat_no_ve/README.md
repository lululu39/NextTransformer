# Nanochat four-stream mHC without attention value embeddings

This ablation inherits the [nanochat-style routing and learned readout](../mhc_nanochat_readout/README.md). Attention values come from the contextual V projection alone. The twelve attention VE tables and the corresponding gates are removed; learnable input smear stays active. The ordinary native residual/x0 scales and midpoint backout are frozen as in the residual family.

Use the matched-normal research launcher. It obtains the common weights from the VE-on model at the same shape before drawing the residual output projections with seed 43. Router draw, initialization and final head follow the nanochat mHC equations implemented on the NextTransformer stack. The model starts from the seed-42 research backbone, and reports step-matched validation against the no-VE Standard and Chord runs. This is a large-model test of their relative behavior; it uses ClimbMix and the NextTransformer schedule.
