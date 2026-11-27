# Instructions for coding agents

NextTransformer is built to be worked on by AI agents as well as people. If you are an agent working in this
repository, follow these rules in addition to the user's request.

## The contract

All methods train under one fixed protocol: the same data, data order, tokenizer, 2,048-token context,
49.152B input tokens, 1,048,576 tokens per update, 46,875 updates, seed 42, Muon + AdamW schedule, 189 validation
events and six full CORE-22 events. Training FLOPs per input token must stay within 1% of the SSSL anchor at
admission and within 2% cumulatively. Never change any of these to help one method. The full contract is in the
[protocol documentation](https://interactive-active-ai.github.io/NextTransformer/docs/protocol/overview/).

## Adding a method

- Add it as a plugin under `nexttransformer/methods/<method>/` ([plugin guide](https://interactive-active-ai.github.io/NextTransformer/docs/guides/plugins/)).
  Do not edit shared model files, the trainer, evaluation, data pipeline, cost ledger, `nexttransformer/models/backbone.py`
  or `nanochat/`.
- Calibrate it with `python -m nexttransformer.protocol.calibrate --methods <method> ...` and keep
  `python -m nexttransformer.protocol.fingerprint` passing.
- Select candidates with training loss or the dev split (`shard_06541.parquet`). The validation shard and CORE-22
  are reserved for the formal run.

## Chord research defaults on this fork

- Default new Chord runs to ambient velocity. Keep the position on the
  unit-RMS sphere, read the lookahead with the actual RMS of h + μv, and
  split the full position displacement along the incoming ambient velocity.
  Keep its radial component in the velocity write.
- Use [token lookup](configs/residual/chordres_ambient_lookup.json) by
  default, with the independent zero-initialized table in the native token
  embedding AdamW group. The [projected-h0 option](configs/residual/chordres_ambient_projected_h0.json)
  uses a shared zero-initialized native Muon matrix on normalized post-smear
  h0. The [zero-start option](configs/residual/chordres_ambient_zero.json)
  starts every token at rest. All three use the existing soft force at τ=0.05
  and six-control update.
- Use six h-only, token-dependent controls as the default for new Chord runs:
  lookahead μ, parallel and perpendicular position steps, old-velocity
  retention β, and parallel and perpendicular velocity-write gains. The final
  sublayer only consumes the read and position controls. Use static controls
  for explicit ablations.
- Use native zero attention output and MLP down projections, h-only dynamic
  matrices initialized at zero, and three learnable modulation scales per
  consumed site starting at 0.01. Preserve the native attention value
  embeddings and smear. For new default runs, use the native
  `nexttransformer.train` optimizer: static biases, h-only projection weights and
  modulation scales share auxiliary AdamW with base LR 0.0006, or
  0.000848528137423857 peak at the registered global batch. The historical
  research control `2574b7dc` used static peak 0.0024494897427831783.
- Keep named controls, historical runs, their initialization, and checkpoint
  configs intact. Document protocol exceptions separately for upstream submissions.

## Running things

- `python -m nexttransformer.train` runs in research mode by default; leaderboard runs need `--formal` and an
  admitted configuration. Research runs are never published.
- Bounded checks only, unless the user explicitly asks for a full training run. Check GPU occupancy first and do not
  disturb other jobs, data, checkpoints or caches.
- Before a pull request: `python -m nexttransformer.checks.changes --base origin/main` and `python -m pytest -q -n auto`.

## Arena entries

If you were given the [arena task](https://interactive-active-ai.github.io/NextTransformer/docs/arena/task/), the
[arena rules](https://interactive-active-ai.github.io/NextTransformer/docs/arena/rules/) also bind you. Record every
candidate, formal launch and human intervention truthfully in the disclosure card.

## Hygiene

- Code, comments and documentation in English.
- Disposable scripts and logs go in `.agent-work/` (ignored). Never commit credentials, datasets, checkpoints, caches,
  machine names or absolute local paths.
- Do not rewrite published results, audits or cost proofs.
