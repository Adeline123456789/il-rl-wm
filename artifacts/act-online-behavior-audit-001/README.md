# T2/T3/T4 actual training replay audit

CPU extraction from each completed, SHA256-verified fixed-final checkpoint. Each portable `transitions.npz` contains the recorded 262144 real transitions' actions, reward, source code, episode ID, timestep, pre-action grasp indicator, object displacement and next-state goal distance. It is a derived audit dataset, not a full replay or resumable checkpoint. No teacher labels, validation/test states or new environment interactions are added.

`scripts/diagnose_behavior.py` checks the PickCube state layout against geometric offsets before extraction. Every 8192-transition bin is reconstructed from the portable arrays and compared with the original extraction. `source.json` identifies the original checkpoint and hash. `counts.json` additionally reports the full-segment counts; `recomputed-counts.json` is the independent file round-trip check.

```bash
CUDA_VISIBLE_DEVICES='' .venv/bin/python artifacts/act-online-behavior-audit-001/recompute.py \
  --input artifacts/act-online-behavior-audit-001 --output /tmp/act-behavior-counts.json
```

Use a new output filename. This CPU command requires the project's NumPy/Torch environment and does not construct a simulator or policy.

Important interpretation limits:

- All source codes are 3, the existing bare-SAC collector route. Its first4096 actions are uniform random warmup, not Actor samples. The first8192 bin mixes these with post-update policy samples.
- The 1312 observed episode IDs include16 partial tail episodes; prior full-episode reports count1296. These denominators must not be interchanged.
- Grasp counts use the saved **pre-action** observation, not a new event detector or post-action terminal flag. They do not establish which action caused a grasp.
- Observed action standard deviation is dispersion across states and random draws, not conditional policy entropy or useful exploration coverage.
- The locked collector performs SAC updates after the4096-transition warmup, before its first learned-Actor action. This common setting is documented as a competing explanation, not changed here or identified as the cause of failure.
