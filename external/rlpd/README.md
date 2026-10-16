# Isolated RLPD comparison preparation

The main learner remains ManiSkill's PyTorch BC/SAC. This directory is a separate JAX comparison using the rfcl implementation named by the locked ManiSkill RLPD instructions. Dependency checks, CPU learner updates, and a real 4096-interaction GPU training smoke pass. Full training is queued; see research_state.json for current status.

rfcl commit: `7bd660f05dbda1ec8881bcfb250a8d1881532dbb`. The documented `train_ms3.py` and `configs/base_rlpd_ms3.yml` are absent from its current tree and the default branch's history for those paths. Its current `train.py` implements reverse/forward curricula. Our entry calls upstream `SAC.update_parameters`, `ActorCritic`, `GenericBuffer`, and `ReplayDataset` directly, without curriculum resets, demonstration action scaling, or validation leakage. The original RLPD source is locked at `c90fd4baf28c9c9ef40a81460a2e395092844f88`, MIT (`LICENCE`). File hashes and checked sources are in source_manifest.json.

The configuration follows RLPD's symmetric sampling, critic LayerNorm, ten critics, and high UTD, with sparse Adroit's three-layer/no-entropy-backup settings as a declared starting point. See the [paper, §4 and appendix B](https://proceedings.mlr.press/v202/ball23a/ball23a.pdf) and the [original author configuration](https://github.com/ikostrikov/rlpd/blob/c90fd4baf28c9c9ef40a81460a2e395092844f88/configs/rlpd_config.py). This is a ManiSkill adaptation, not an exact reproduction of the paper's environments or the missing ManiSkill configuration. protocol.json lists initialization, log-standard-deviation, collection and warmup deviations. Its larger critic/update budget is reported separately from C/U0; it is not an equal-compute comparison.

The upstream GPU GymLoop leaves final-observation handling as `pass`. Our small collector uses the real final observation before automatic reset and preserves raw labels, with continuing-task bootstrap. It does not change upstream loss functions. The CPU check verifies the final/reset distinction, 50% alternation, ten-critic output, all three LayerNorm layers, 20 critic updates per actor update, exact optimizer serialization, and the 717 training episodes' raw reward labels. No new learner is imported into the main PyTorch path.

Real smoke command (after the current GPU queue, under its shared lock):

```sh
python scripts/run_experiment.py --id e3-rlpd-smoke-001 --seconds 1800 -- .venv-rlpd/bin/python external/rlpd/train.py --output artifacts/e3-rlpd-smoke-001 --exp-name e3-rlpd-smoke-001 --seed 0 --smoke
```

The smoke has 4096 real interactions, 1024 random warmup interactions and 61440 critic updates. Full 524288-interaction training (262144 intermediate checkpoint) is gated on smoke correctness and measured runtime, matching U0/C continuation total online interactions including their shared preheat. Artifacts include every validation episode, training episode, periodic learner metrics, final-observation checks, real interaction counts and manifests. Checkpoints include actor/ten critics/targets/optimizers/temperature, RNG, replay and raw labels. They do not serialize the simulator; resumption is a new segment.

Dependencies are isolated in `.venv-rlpd`; requirements here do not change `.venv` or the main dependency lock. CUDA imports during installation checks must use `JAX_PLATFORMS=cpu` so they do not compete with active main experiments. Real GPU tests must use the shared bounded experiment runner, with `XLA_PYTHON_CLIENT_PREALLOCATE=false`.

The rfcl package metadata declares MIT, but the inspected tree lacks a separate LICENSE file. No rfcl source is copied into this deliverable. Source commit/file manifests are retained, and the cache checkout remains unchanged.

## Portable actor and independent evaluation

`policy.py` exports only the locked rfcl actor parameters and protocol. It never constructs critics or optimizers for inference. The smoke export is 578938 bytes; 64 actual replay observations produced bit-identical actions against the complete saved learner on CPU (`artifacts/rlpd-policy-export-check-001`). This is inference parity, not final policy performance.

```bash
.venv-rlpd/bin/python external/rlpd/policy.py --checkpoint checkpoints/e3-rlpd-full-001/final --output releases/rlpd-full-001
.venv/bin/python scripts/start_experiment.py --id e7-rlpd-test-001 --seconds 300 -- .venv-rlpd/bin/python external/rlpd/evaluate.py --policy releases/rlpd-full-001 --output artifacts/e7-rlpd-test-001 --seed 20000 --episodes 100 --num-envs 10
```

Run the final-test command only after full training and registered validation smoke. The evaluator uses the same whole-episode helper, reward, horizon, continuing semantics and independent seeds as main SAC; it performs no learning and records raw episodes/interactions. The independent loader GPU smoke completed 4 validation episodes (800 interactions, no learning), matching the registered state/sparse/continuing protocol; this is not the final 100-episode test. The exported actor cannot resume training; the complete checkpoint remains separately archived.
