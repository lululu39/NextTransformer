# Velocity ablations and early residual dynamics

The [token-velocity run](https://wandb.ai/yibozhong657-none/nexttransformer-chordres/runs/7228d8cf) stopped at its complete step-500 checkpoint. It had used 524,288,000 training tokens. All interventions below are evaluation-only. The [matched zero-velocity run](https://wandb.ai/yibozhong657-none/nexttransformer-chordres/runs/1879c1a2) supplies the 500-step training comparison and the 8,000-step lookahead test.

## Full validation BPB

Eight H100s replayed the official 41,943,040-token validation set, at batch 8 per GPU and 2,048 context. Each unmodified checkpoint reproduced its logged BPB. Interventions preserve all other weights and use the same token order.

| Checkpoint | Intervention | BPB | Change from intact |
|---|---|---:|---:|
| Token velocity, 500 | Intact | 0.92629674 | +0.00000000 |
| Token velocity, 500 | Rotate vocabulary rows by 7,919 | 0.92676889 | +0.00047215 |
| Token velocity, 500 | Zero the initial velocity table | 0.92679136 | +0.00049462 |
| From rest, 8,000 | Intact | 0.78226884 | +0.00000000 |
| From rest, 8,000 | Zero every lookahead μ | 1.03851505 | +0.25624622 |

The independently trained, from-rest control reached 0.92669327 at step 500. Rotating the learned table preserves all rows and changes which token retrieves each row. The 8,000-step intervention disables the lookahead read while retaining velocity in the write. These values measure sensitivity of trained weights to an inference intervention.

## Residual geometry

Each model processed the same 24 validation-text excerpts (first 4,096 characters of each selected ClimbMix document), using its own tokenizer. Ratios exclude BOS and right padding. Initialization reconstructions use seed 42 and each model's initialization recipe. FLAME ChordRes and mHC also use their original two-update distributed preflight checkpoints. The diagnostic compares fixed early weights on common short text, rather than replaying the training trajectory at 32K. For mHC the measured hidden is the arithmetic mean across streams. Each measurement spans a whole Transformer block (attention and FFN).

The relative update is RMS(h_out - h_in) divided by RMS(h_in), pooled over all measured tokens and channels. The final cosine is the mean tokenwise cosine of the last block's hidden with the initial hidden before the final norm. Here 'normal' for NextTransformer means the same depth-scaled random output projections used by the completed ChordRes. 'Native zero' refers to zero output projections at initialization.

| Model or frozen intervention | Initial hidden RMS | Block 1 relative update | Last hidden cosine to input |
|---|---:|---:|---:|
| FLAME Standard, initial | 0.0200 | 22.4578 | 0.0076 |
| FLAME mHC, initial | 0.0200 | 22.4220 | 0.0076 |
| FLAME ChordRes, initial | 1.0000 | 0.0743 | 0.9351 |
| FLAME mHC, actual step 2 | 0.0200 | 22.4659 | 0.0090 |
| FLAME ChordRes, actual step 2 | 1.0000 | 0.0743 | 0.9356 |
| NextTransformer Standard, matched normal | 1.0000 | 0.0523 | 0.9692 |
| NextTransformer ChordRes, matched normal | 1.0000 | 0.0744 | 0.9384 |
| NextTransformer Standard, native zero | 1.0000 | 0.0000 | 1.0000 |
| NextTransformer mHC, native zero | 1.0000 | 0.0000 | 1.0000 |
| FLAME Standard, unit-RMS input only | 1.0000 | 0.4529 | 0.4077 |
| FLAME Standard, small output projections only | 0.0200 | 3.2571 | 0.0627 |
| FLAME Standard, both changes | 1.0000 | 0.0654 | 0.9523 |
| NextTransformer Standard, input RMS 0.02 and large output projections | 0.0200 | 17.8019 | 0.0037 |

The FLAME Standard intervention rescales the same initial attention output and FFN down projection weights by 1/sqrt(48), normalizes only the incoming embedding residual, or applies both. The reverse NextTransformer intervention multiplies its matched projections by sqrt(46) and scales the embedding residual by 0.02. These forward-only comparisons isolate effects on geometry. Establishing effects on learning curves would require training a matched intervention.

[Per-layer measurements, sharding-independent validation scores and text selection](artifacts/chordres-v-early-20261123.json)
