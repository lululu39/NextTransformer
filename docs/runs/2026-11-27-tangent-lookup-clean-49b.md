# Project-only Chord with token lookup from step zero

The owner selected simple tangent projection as the default velocity geometry. This run uses the same [calibrated lookup configuration](../../configs/residual/chordres_soft_force_token_velocity.json) and frozen source revision `a9220df6a4d4` as the [earlier 5,000-step tangent lookup run](https://wandb.ai/yibozhong657-none/nexttransformer-chordres/runs/3d1ac486). The model configuration, native trainer entry and uv lock were checked byte for byte against that run. The ambient ablation stopped at step 3,000 with its full checkpoint retained.

Start independently at step zero with seed 42, an empty optimizer and the beginning of the ClimbMix stream. Use eight H100s, context 2,048, per-rank microbatch 8 and eight accumulation steps. The goal is 46,875 updates (49.152B tokens) with the original evaluation and LR schedules. Each token starts with a tangent projection of its zero-initialized velocity lookup. Each sublayer uses six h-conditioned scalars and the existing soft-force update. The velocity table shares the input embedding AdamW group (peak 0.30), and the controls share the official auxiliary AdamW group (peak 0.000848528137). The model has 1,406,561,954 parameters and costs 4.687104 GFLOPs per token. Production uses `--resume never`, retains rolling checkpoints and stops on interruption. The [completed 49.152B zero-start control](https://wandb.ai/yibozhong657-none/nexttransformer-chordres/runs/61fdbef5) supplies the matched zero-initial-velocity endpoint.

## Verified launch

The eight-rank preflight completed three full-batch updates. All eight production ranks verified fresh optimizer groups. The [new run](https://wandb.ai/yibozhong657-none/nexttransformer-chordres/runs/3ac13380) matched the original seed-42 control's step-zero BPB (3.16874171) and first production LM loss (10.39769365).

## Effective lookahead at update 500

All 46 sublayers learned nonzero static lookahead biases and nonzero h-dependent lookahead rows. On 131,072 ClimbMix validation tokens the effective μ had a positive RMS and nonzero token-wise standard deviation at all 46 sites. First attention had bias -0.000271, effective μ RMS 0.07674, and RMS(μv)/RMS(h) of 40.09%. The median sublayer offset ratio was 36.67% (range 2.77–67.40%). The archived 5k lookup control also had nonzero biases and dynamic lookahead rows at all 46 sites. The complete fixed-zero-velocity control had 45 such sites, with a structurally unused first attention lookahead. [Per-sublayer results](artifacts/chordres-clean-lookup-step500-mu.json)

## Velocity-table use at update 35,000

On 64 held-out ClimbMix sequences (131,072 next-token targets), the lookup model scored 2.50887 nats per token. Zeroing its velocity table during evaluation raised CE to 3.34669. Shuffling its lookup indices across the same tokens raised CE to 2.79871. Both interventions increased mean CE for all 64 sequences. The independently trained fixed-zero model at step 35,000 scored 2.51071 on those same inputs. On the 41,943,040-token periodic validation, lookup improved BPB by 0.000332 at step 35,000 and 0.000287 at step 38,000. [Paired diagnostic](artifacts/chordres-lookup-vs-zero-35k.json)
