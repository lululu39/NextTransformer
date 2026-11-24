# VE-specific velocity steering in the three-scalar h-only model

This tests the [observed decrease in velocity/displacement alignment](2026-11-24-honly-ve-velocity-alignment.md) on frozen [h-only Chord checkpoints](https://wandb.ai/yibozhong657-none/nexttransformer-chordres/runs/345e22ad) from updates 1,000 and 1,500. Only VE-enabled attention writes with a defined incoming velocity are modified (11 layers). The original training job runs separately. The native position write, branch, and static model weights are retained. Subsequent tokens and layers in the diagnostic see the intervened trajectories.

The direct local signal is the difference in cos(v, h_next − h) between VE-on and VE-off attention outputs evaluated on identical inputs, with the intact attention output written to the state. Original read controls, attention input, Q, K and attention window are shared across the pair. The first test adds that signal to the beta logit: when VE lowers alignment, beta decreases and the write puts more weight on the new displacement. The opposite sign, a within-sequence permutation of the token signal, and its calibration-set layer mean provide controls. A cheaper alternative uses 0.5 times the centered intact cosine alone. All interventions preserve beta in [0,1].

The second test preserves the length of the normally written velocity and changes only its direction. At each VE site take the VE-specific displacement difference h_next_on − h_next_off, project to the new tangent plane and remove the component parallel to the outgoing tangent velocity. Add ±0.5 times this vector to the outgoing velocity and restore its original norm. An additional run caps each local rotation at 3 degrees. A randomized-direction control preserves the proposed turn length on each token before the speed-preserving update. Angles in the table refer to the VE-ward intervention. Undefined directions are left at the native update.

All tests use the separate ClimbMix development shard 06541 (eight fixed packing streams). Calibrate layer averages on the first 65,536 tokens, then score the next 524,288 tokens, with the same 256 sequences for each condition. Native BPB excludes special targets. Masking off every intervention exactly reproduces the original per-token losses on the first two sequences per rank. Row-bootstrap intervals resample the 256 sequences together for each paired contrast. The measurements evaluate frozen-checkpoint sensitivity; the effect of training with a learned steering parameter is unmeasured.

| Intervention | ΔBPB at 1,000 | ΔBPB at 1,500 |
| --- | ---: | ---: |
| Use VE-induced alignment change in beta | -0.0000072 | +0.0000006 |
| Opposite beta response | +0.0000007 | +0.0000012 |
| Permute VE signal across tokens | +0.0000070 | +0.0000042 |
| Use per-layer mean VE signal | -0.0000006 | +0.0000125 |
| Use only intact angle in beta | +0.0000057 | +0.0000110 |
| Rotate toward VE-specific displacement (at most 3°) | -0.0000113 | +0.0000751 |
| Rotate away (at most 3°) | +0.0001987 | +0.0001646 |
| Rotate toward randomized feature direction (at most 3°) | +0.0000614 | +0.0000818 |

| Rotate toward VE-specific displacement (uncapped) | +0.0069759 | +0.0081761 |
| Rotate away (uncapped) | +0.0071431 | +0.0073324 |
| Rotate toward randomized feature direction (uncapped) | +0.0029191 | +0.0029533 |

Mean absolute VE-ward rotation across non-special token/site pairs was 3.00° (capped) / 22.44° (uncapped) at 1000, 2.99° (capped) / 21.58° (uncapped) at 1500.

An earlier separate [from-scratch static VE-turn run](2026-11-24-chordres-ve-turn.md) retained the full perpendicular step at VE writes and had validation BPB 0.914588 versus 0.915757 at update 500 for its matched static control. This historical rule changes velocity magnitude as well as direction and used a different Chord model.

[Paired measurements, uncertainty intervals, sequence fingerprints and speed checks](artifacts/honly-ve-turn-adaptation.json).
