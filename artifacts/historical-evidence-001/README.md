# Reconstruct original evidence

Nine parts reconstruct raw-evidence.tar.gz (34020522 bytes), SHA256 b502cd919c85274e247f122c191b3ea1b12ef1e3e723a9210cf9fea45e2df3ae. Individual part hashes and every original file hash are in inventory.json. The full local tar is deliberately not duplicated in Git.

From repository root, reconstruct into a new directory to preserve any existing experiment artifacts:

```sh
mkdir -p .cache/historical-evidence-recompute
cat artifacts/historical-evidence-001/raw-evidence.tar.gz.part* > .cache/historical-evidence-recompute/raw-evidence.tar.gz
sha256sum .cache/historical-evidence-recompute/raw-evidence.tar.gz
tar -xzf .cache/historical-evidence-recompute/raw-evidence.tar.gz -C .cache/historical-evidence-recompute
python scripts/summarize_historical_validation.py --root .cache/historical-evidence-recompute/artifacts --output .cache/historical-validation-recomputed
```

The summary uses only Python standard library. Choose an unused output directory for repeated reproduction. Compare conditions/aggregates/evidence with artifacts/historical-validation-001/summary.json (the CLI also adds plan_sha256). No model execution, GPU or final-test access occurs.
