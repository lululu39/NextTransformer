# Direct VE impulse on the static, matched-normal Chord control

The paired control is [13fb5703](https://wandb.ai/yibozhong657-none/nexttransformer-chordres/runs/13fb5703), with two positive scalar position steps and complementary single-beta velocity mixing. Its step-2,000 validation BPB is 0.81365203. Train the same model from rest on the same data, with one extra signed scalar shared across all 12 attention VE sites. The embedding is the raw same-token lookup already used by attention. The scalar starts at zero and receives the residual controls' AdamW peak LR 0.0024494897427831783. The new term is added to velocity before projection at the newly updated position.

Keep seed 42, random branch output projection seed 43, 23 blocks, width 1536, FFN 6144, BF16 saved forces, the ordinary LM head, smear and the original attention VE gates. All 46 sites retain their static Chord controls. Use the established eight-rank ClimbMix stream, 2,048 context, B8 per rank and eight accumulation steps. The 2,000-update research segment uses the original 46,875-update LR schedule. Compare BPB every 250 updates and log the learned shared gain. Retain the end checkpoint and the standard gradient-stop rule.

The admitted model has 1,355,810,752 parameters, one beyond its paired control. Theoretical training cost is 4.680948 GFLOPs per token (0.183% above the compute anchor).
