# Token-dependent ChordRes, 8,000 updates

The paired static model is the [three-gain ChordRes run](https://wandb.ai/yibozhong657-none/nexttransformer-chordres/runs/a710a057). It stopped after writing its 2,500- and 3,000-update checkpoints. Compare validation BPB every 250 updates through the common 3,000-step window.

Train from scratch using the [token controls](../../nexttransformer/methods/chordres_token_controls/) on the fixed 23-layer, width-1,536, FFN-6,144 shell. The 838,656 new weights use the existing Chord AdamW group at peak LR 0.0024494897427831783, with zero weight decay. The static biases and all shared weights retain their original initialization. Use seed 42 and seed-43 normal attention and MLP output projections, zero initial velocity, 12 attention value embeddings, input smear, saved BF16 forces and the ordinary LM head.

Train on the original ClimbMix stream at 2,048 context, B8 per rank, eight ranks and eight accumulation steps. Stop after 8,000 updates, or 8,388,608,000 tokens, of the original 46,875-update schedule. Evaluate validation BPB every 250 steps and CORE-22 at 8,000. Record the dispersion of each learned control on the first validation batch at rank zero. The comparison uses the same W&B entity and project as the static run.
