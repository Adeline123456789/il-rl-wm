"""Summarize existing measurements; no inference, training, or simulation."""
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

out = Path('artifacts/seed0-runtime-costs-001')
sources, summaries, evaluations, timing_rows = {}, [], [], []
for variant, name in [('U0', 'e6-u0-full-001'), ('U1', 'u1-pick-seed0-u0-001')]:
    folder = Path('artifacts') / name

    def read(filename, lines=False):
        path = folder / filename
        data = path.read_bytes()
        sources[str(path)] = hashlib.sha256(data).hexdigest()
        return [json.loads(line) for line in data.splitlines()] if lines else json.loads(data)

    manifest = read('manifest.json')
    assert manifest['exit_code'] == 0 and manifest['stop_reason'] is None
    result, model = read('result.json'), read('model.json')
    timings, resources = read('planning.jsonl', True), read('resources.jsonl', True)
    assert len(timings) == 256
    assert [r['step'] for r in timings] == list(range(1024, 262145, 1024))
    assert all(r['batch_size'] == 16 for r in timings)
    latency = np.array([r['latency_ms'] for r in timings])
    timing_rows.extend(dict(variant=variant, **{k: row[k] for k in ['step', 'batch_size', 'latency_ms']}) for row in timings)
    costs = read('evaluation_cost.jsonl', True)
    assert len(costs) == 20 and sum(r['environment_interactions'] for r in costs) == result['validation_interactions']
    evaluations.extend(dict(variant=variant, **row) for row in costs)
    summaries.append(dict(
        variant=variant, experiment=name, model=model,
        new_training_interactions=result['training_interactions'],
        lineage_training_interactions=result['inherited_training_interactions'] + result['training_interactions'],
        validation_interactions=result['validation_interactions'],
        gpu_job_wall_seconds=manifest['elapsed_seconds'],
        learner_and_validation_wall_seconds=result['elapsed_seconds'],
        measured_action_selection_batches=len(timings), batch_size=16,
        action_selection_ms_mean=float(latency.mean()),
        action_selection_ms_median=float(np.median(latency)),
        action_selection_ms_p95=float(np.percentile(latency, 95)),
        action_selection_ms_max=float(latency.max()),
        sampled_whole_gpu_peak_mib=max(r['memory_mib'] for r in resources),
        sampled_process_peak_rss_kib=max(r['VmHWM_kib'] for r in resources if 'VmHWM_kib' in r),
        torch_peak_allocated_bytes=result['torch_peak_allocated_bytes']))

report = dict(
    summaries=summaries, evaluation_costs=evaluations, sources=sources,
    new_environment_interactions=0,
    measurement_scope='Action selection includes H2 candidate generation, task-value scoring, exploration sampling and source counters, with CUDA synchronization; excludes env.step, gradient updates and JSON serialization. Evaluation wall time includes simulator steps and episode bookkeeping.',
    limitations='One development training run per variant. Sparse timing samples every 1024 transitions, not an independent hardware benchmark. Quantiles are descriptive, not confidence intervals. Resource sampling every two seconds can miss transient peaks. These training samples use IL and batch16; they are not standalone no-IL or batch1 deployment latency.')
(out / 'result.json').write_text(json.dumps(report, indent=2) + '\n')
for filename, rows in [('action_selection.csv', timing_rows), ('evaluation_costs.csv', evaluations)]:
    with (out / filename).open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)
