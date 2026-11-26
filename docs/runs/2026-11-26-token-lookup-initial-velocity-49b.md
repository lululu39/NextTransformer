# Token lookup for the initial Chord velocity

## Full-budget run selection

On 2026-11-26, token lookup was selected for the learned-initial-velocity run through update 46,875 (49.152B tokens). Continue the existing run `3761f83a` with its six h-conditioned scalars and original schedule. The completed zero-velocity run provides the full-budget control. The projected-h0 run retains its step-2,500 checkpoint for reference. Further comparison of that initialization is deferred to smaller models.

At the time of selection, lookup had lower periodic validation BPB at each of the nine shared evaluations from steps 250 to 2,250. At step 2,250 the scores were 0.806647 for lookup and 0.807065 for projected h0.

### Matched 5,000-step comparison

The lookup run paused at update 5,000 on November 26 after validation at 0.78628687 BPB. The retained checkpoint contains model weights and all eight rank shards, including optimizer state, RNG and data cursors. The [projected-h0 continuation](https://wandb.ai/yibozhong657-none/nexttransformer-chordres/runs/e1c681f3) resumed from its original update-2,500 checkpoint with the same training configuration, software and 46,875-step learning-rate schedule. W&B linked it to its [original run](https://wandb.ai/yibozhong657-none/nexttransformer-chordres/runs/58d95ca4), which had logged unsaved iterations through 2,690. Both variants use the same 41,943,040-token periodic validation at every 250 updates. The [original projected-h0 run](https://wandb.ai/yibozhong657-none/nexttransformer-chordres/runs/58d95ca4) also shows the continuous comparison curve. It retains the original training points through 2,690 and appends the resumed training and validation events from update 2,700. The native continuation retains the full replay from 2,500.

## Training configuration

The learned initial velocity comes from an untied vocabulary table instead of the projected-h0 control's shared linear map. Start all rows at zero and project the lookup onto the tangent plane at the same normalized post-smear position. Use the native token-embedding AdamW settings (peak 0.30, betas 0.8 and 0.995, weight decay 0.001). The lookup contains 50,331,648 parameters. Shared model parameters, initialization, native auxiliary learning rates and the subsequent Chord updates follow the projected-h0 experiment.

Train the [calibrated configuration](../../configs/residual/chordres_soft_force_token_velocity.json) from scratch with seed 42 for 46,875 updates on 49.152B ClimbMix tokens. Use BF16, 2,048 context, 1,048,576 tokens per update, eight H100s, microbatch eight and eight accumulation steps. The input velocity, Chord position and branch states retain BF16. Keep attention value embeddings, soft force with tau 0.05, 40-step warmup, checkpoints, periodic validation and the full CORE-22 schedule. Dependencies use the same uv.lock as the two paired runs.

The [compute ledger](../../evidence/costs/chordres-soft-force-token-velocity-v1/chordres_soft_force_token_velocity.json) gives 1,406,561,954 parameters and 4.687104 GFLOPs per token, 0.3147% above the reference. The projected-h0 control was stopped at its owner's request with a complete step-2,500 checkpoint. Its traces and the completed zero-velocity run are available in the same project. [Projected-h0 control](https://wandb.ai/yibozhong657-none/nexttransformer-chordres/runs/58d95ca4) · [Zero-velocity control](https://wandb.ai/yibozhong657-none/nexttransformer-chordres/runs/61fdbef5)

The eight-rank preflight matched all 443 shared initial tensors and the zero-velocity control logits. It completed three full-batch forward and backward updates, with a nonzero velocity-table gradient on update three. All eight production ranks verified the native embedding and auxiliary optimizer groups. The [full training run](https://wandb.ai/yibozhong657-none/nexttransformer-chordres/runs/3761f83a) started from step zero at revision `a9220df6a4d4`.

## Initial velocity at update 500

On 64 validation sequences (131,072 next-token targets), held-weight cross-entropy was 3.18490 nats/token. Clearing only the initial velocity raised it by 0.02207 (paired sequence-bootstrap 95% interval 0.02034–0.02387). Permuting the velocity lookup token IDs while preserving their frequencies and the backbone inputs raised it by 0.01351 (0.01231–0.01469). Both interventions increased mean CE for all 64 sequences. The token-weighted projected velocity RMS was 5.4392, the first effective lookahead coefficient RMS was 0.07253 and the first lookahead offset RMS was 38.02% of the position RMS. On the separate 41,943,040-token periodic validation, the independently trained lookup and fixed-zero models scored 0.902747 and 0.903479 BPB, respectively. [Paired diagnostic measurements](artifacts/chordres-token-lookup-initial-velocity-step500.json)
