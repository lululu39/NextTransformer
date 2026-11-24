# Velocity carry across VE attention layers

Frozen checkpoints 1,000 and 1,500 from [the three-scalar h-only Chord](https://wandb.ai/yibozhong657-none/nexttransformer-chordres/runs/345e22ad) were evaluated on the same 256 development sequences as the [steering probe](2026-11-24-honly-ve-turn-adaptation.md), totaling 524,288 input tokens. Each sequence pair shares token IDs, targets and packing order. The intact per-token losses and baseline BPB were verified against the earlier test. The input velocity at the first attention is zero.

The proposed rule keeps the ordinary VE-on position write at all sites. At all 12 attention sites with VE it sets effective velocity retention to 1: v_next = P_at(h_next)(v). All MLPs and all attention sites without VE use the trained update. Each later branch reads its intervened trajectory. The custom projection-only write matched the native fused kernel with beta=1 on the checked batches. Position outputs also matched the native calculation at fixed inputs.

A separate local control computes a second attention on the same input and Q/K with only the VE contribution to V suppressed, then computes its hypothetical new position. It uses the original beta to write that ordinary-attention displacement into velocity, projected at the actual VE-on position. This removes the local VE contribution to the displacement while preserving the ordinary attention write and its retention. For the site-count comparison, omit the first VE attention, leaving 11 odd-positioned VE attention sites versus 11 even-positioned non-VE attention sites. Their identities and depths also differ.

| Velocity intervention | Δ dev BPB, update 1,000 | Δ dev BPB, update 1,500 |
| --- | ---: | ---: |
| Project old velocity only at every VE attention (proposed) | +0.027320 | +0.034505 |
| Keep decay and remove all displacement at VE attentions | +0.036569 | +0.060473 |
| Keep the displacement write and remove decay on old velocity | +0.005450 | +0.006417 |
| Write the VE-off displacement at the VE-on position | +0.001929 | +0.001757 |
| Project old velocity only at the first attention | +0.006796 | +0.008355 |
| Project old velocity at the 11 later VE attentions | +0.016245 | +0.021801 |
| Project old velocity at the 11 non-VE attentions | +0.940483 | +1.059410 |

Baseline BPB was 0.841287 at 1000, 0.817609 at 1500. The proposed projection-only rule increased loss on every one of the 256 sequences at both snapshots. Intervals and directional contrasts resample whole packed sequences (5,000 draws). These results measure use of the velocity path by weights trained with ordinary writes. Training from scratch with a changed recurrence is a separate experiment. [Complete paired results](artifacts/honly-ve-projection-only.json).
