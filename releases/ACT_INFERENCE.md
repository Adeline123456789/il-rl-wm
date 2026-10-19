# ACT comparison inference packages

`scripts/act_inference.py` loads the existing teacher (`kind=teacher`), distilled
MLP (`mlp`), or independently transferred Transformer (`act`) with strict tensor
matching. Student loading never constructs an ACT teacher. Learned stochastic
variance is loaded from the actor checkpoint rather than reset to its initializer.
All inference parameters are frozen; these packages omit replay and optimizers
and cannot resume training.

`act-t2-online-final-001/weights.pt` contains the completed random-SAC control's
actor (584114 bytes). The export manifest links its SHA256 to the original full
training checkpoint and verifies every actor tensor plus deterministic CPU
serialization parity. This CPU check adds no robot interaction and is not a new
success-rate measurement. T2's existing fixed-final validation is 0/40.

Re-export a completed no-planning online condition with:

```sh
.venv/bin/python scripts/export_act_online_actor.py \
  --checkpoint checkpoints/act-t2-online-seed0-002/final.pt \
  --output releases/NEW_OUTPUT_ID
```

Run this through a detached CPU job with `CUDA_VISIBLE_DEVICES=''`, recording its
PID, command, log and exit code, as in `artifacts/act-t2-export-001`. A fresh output
directory is required. The original final-checkpoint hash and completed run are
checked before exporting.

`scripts/evaluate_act_checkpoint.py` is the independent closed-loop entry point:
teacher modes `first`/`ensemble`; student modes `mean`/`stochastic`. It retains
raw state/action/reward traces, episode seeds/events/returns and measured latency.
Validation seed10000 and final seed20000 are separate; final requires 100 episodes,
10 parallel environments, explicit seeded resets for every batch and the existing
released final-test gate. CPU gate/loader checks passed; the new 10-environment
entry has **not yet run in real simulation**. Do not call it directly during the
online queue: every future simulation must use the single-GPU budget runner.

This entry neither registers new final conditions nor releases the gate. Freeze
the comparison and checkpoint selections first, account for the existing 88
historical conditions and ACT conditions inside the shared remaining budget, and
retain the 7200-second final-evaluation reserve. Student-only SAC inference here
does not establish the final SAC+WM research objective.

T3 BC-initialized SAC: `act-t3-online-final-001/weights.pt` (584178 bytes),
exported after its successful full-run exit. Its manifest and
`artifacts/act-t3-postprocess-001` preserve full-source hash, strict tensor parity
and CPU serialization checks. Existing final validation remains 0/40.
