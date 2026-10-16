# Recompute the U0 failure evidence

`u0-analysis-precollapse-001/weights.pt` and `u0-analysis-final-001/weights.pt` preserve model tensors from the 196608-step and fixed-final U0 snapshots. Each manifest maps the export to the original checkpoint SHA256. These **research-only files include frozen IL**; they are separate from the IL-free `u0-final-001` deployment package. No replay, optimizer or RNG is included; they cannot resume training.

The fixed real inputs are in `artifacts/f-u0-value-drift-002/fixed_batches.npz`: 2048 expert and 2048 online transitions sampled with replacement using seed532. Original sample indices are retained. Expert data comes from the locked official ManiSkill demonstration revision and corrected GPU replay; online inputs come from the 196608-step U0 replay. See the original experiment expert_manifest.json and dataset-audit.json for the full episode split and dataset hashes. These samples are diagnostic inputs, not additional training data.

From repository root, with the locked main environment:

```sh
CUDA_VISIBLE_DEVICES= .venv/bin/python scripts/diagnose_value_drift.py \
  --before releases/u0-analysis-precollapse-001/weights.pt \
  --after releases/u0-analysis-final-001/weights.pt \
  --fixed-batches artifacts/f-u0-value-drift-002/fixed_batches.npz \
  --output artifacts/local-drift-recompute
```

Use a fresh output directory. This requires no simulator interaction or original large replay files. All saved per-sample arrays matched the original full-checkpoint calculation bit for bit in `artifacts/portable-drift-check-001/result.json`. Bellman residuals are not Monte Carlo calibration or proof of causality.

F1 scoring summaries can be recomputed without GPU from the archived real branch arrays:

```sh
CUDA_VISIBLE_DEVICES= .venv/bin/python scripts/summarize_ranking.py \
  --experiment artifacts/f1-u0-ranking-001
CUDA_VISIBLE_DEVICES= .venv/bin/python scripts/summarize_ranking.py \
  --experiment artifacts/f1-u0-precollapse-ranking-001
```

Both original prefix40 F1 runs completed and their arrays are archived. To repeat simulation branches from a small weight file, use `scripts/diagnose_ranking.py --checkpoint releases/u0-analysis-final-001/weights.pt --output <fresh-path>` through `scripts/start_experiment.py` with a unique ID and a bounded runtime, on the locked single-GPU environment. This consumes diagnostic interactions and is distinct from recomputing the saved evidence. The later prefix20 follow-up has its own registered protocol and status; do not conflate these probes.

U1 seed0 analysis weights and deployment/behavior reproduction are described in [REPRODUCING.md](../REPRODUCING.md). They preserve the same fixed step196608/final snapshot rule; the shared U0 diagnostic inputs and U1-specific replay inputs are reported separately.
