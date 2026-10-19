# Historical BC/U0/U1 validation snapshot — 2026-10-19

This is a recomputation of existing validation logs, not final testing and not ACT evidence. All 52 preregistered checkpoint entries (88 method conditions) have completed training evidence from 51 distinct runs; the two RLPD budgets share one run. Final test seed20000 remains closed. No new environment interactions or GPU inference were used here.

The table uses fixed final checkpoints at 524288 cumulative training interactions. Each cell is success at least once / 16 validation episodes. Seed0 was the development run; seeds1–4 were independent training repeats. Their evaluation episodes do not count as independent training runs.

| Variant / inference | seed0 (development) | seed1 | seed2 | seed3 | seed4 | seeds1–4 mean success |
|---|---:|---:|---:|---:|---:|---:|
| C continuation / sac | 16/16 | 15/16 | 0/16 | 0/16 | 16/16 | 48.438% |
| C-LN continuation / sac | 16/16 | 14/16 | 15/16 | 15/16 | 13/16 | 89.062% |
| U0 / sac | 0/16 | 15/16 | 0/16 | 0/16 | 16/16 | 48.438% |
| U0 / il_sac_wm | 0/16 | 13/16 | 0/16 | 0/16 | 13/16 | 40.625% |
| U0 / sac_wm | 0/16 | 16/16 | 0/16 | 0/16 | 16/16 | 50.000% |
| U1 / sac | 16/16 | 16/16 | 14/16 | 15/16 | 13/16 | 90.625% |
| U1 / il_sac_wm | 13/16 | 12/16 | 9/16 | 11/16 | 9/16 | 64.062% |
| U1 / sac_wm | 15/16 | 15/16 | 14/16 | 16/16 | 13/16 | 90.625% |

The LayerNorm version has higher and less variable success across these four repeats than the historical version. This supports further stability analysis, but does not establish a final-test effect or assign the gain to ACT. U1 no-IL WM and its bare SAC have identical mean success (90.625%) across seeds1–4; the planner therefore has no demonstrated mean success advantage here. With the weak BC candidate present, U1 planning is lower (64.0625%). These are observations, not evidence that the teacher alone causes the deficit. The old stochastic planning evaluations do not establish matched initial states across methods.

StackCube and PegInsertionSide seed0 final A/B/C/continued-C/U0 conditions all have 0/16 success at their registered final checkpoints. Their protocols differ: Stack uses horizon300/pd_ee_delta_pos, Peg horizon400/pd_ee_delta_pose. These bounded negative results remain included, not removed from the task set.

The isolated RLPD development run has 4/16 at262144 interactions and 5/16 at524288. Its full run uses10403840 critic updates,520192 actor updates and6007.349 GPU-job seconds. It shares the demonstration split and sparse task protocol but is not an equal-compute SAC control. Neither RLPD nor the other bounded results proves convergence.

## Reproduction and coverage

`scripts/summarize_historical_validation.py` verifies completed manifests, training/validation interaction counts, task/seed/control/horizon identity, exact checkpoint references, complete episodes and source-file hashes. It emits all88 rows, training-run aggregates with sample standard deviation, and source hashes in `artifacts/historical-validation-001`. It never opens final-test artifacts. Three initial summary attempts failed on historical schema assumptions (missing single-method labels, task-specific horizons/control, and RLPD checkpoint directories); logs are preserved in job001–003. Job004 succeeds after handling the actual registered formats. These corrections do not alter training or original evidence.

`artifacts/historical-evidence-001` contains a chunked archive of1428 original evidence files from82 completed runs including BC/WM/value/resume dependencies. The archive includes raw metrics, per-episode logs, configs, source snapshots and checkpoint manifests. Every archived byte was checked against the inventory; recomputing all88 conditions from a fresh extracted copy produced exactly the original summary. Large checkpoints, full replay, raw demonstration datasets, videos and ongoing ACT online runs are excluded. Thus this package delivers the historical raw numerical evidence, not every training weight or a complete final research result.

Remaining work: ACT online controls; failure-driven bounded comparison if warranted and affordable; final-test gate release after comparison freeze/negative closure; registered final tests and complete result analysis. ACT–WM must still satisfy the transfer gate. The scientific target is unchanged.
