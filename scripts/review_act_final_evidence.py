"""Read frozen final-test evidence; never train, select models, or start simulation."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--summary', type=Path, default=Path('artifacts/act-final-summary-003/summary.json'))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    summary_path = args.summary
    summary = json.loads(summary_path.read_text())
    sources = {str(summary_path): digest(summary_path)}
    rows = []
    reference = None
    initial_actions = {}
    for condition in summary['conditions']:
        if not condition['id'].startswith('act-'):
            continue
        row = dict(condition)
        directory = Path('artifacts') / summary.get('attempt_mapping', {}).get(
            condition['id'], 'e7-test-' + condition['id'] + '-001')
        if condition['status'] != 'completed':
            row['stage_counts'] = None
            row['initial_observation_max_difference'] = None
            rows.append(row)
            continue
        episodes_path = directory / 'episodes.jsonl'
        trace_path = directory / 'final_test_trace.npz'
        records = [json.loads(line) for line in episodes_path.read_text().splitlines()]
        assert len(records) == 100
        assert [r['episode_index'] for r in records] == list(range(100))
        assert [r['eval_seed'] for r in records] == [20000 + 10 * (i // 10) for i in range(100)]
        with np.load(trace_path) as trace:
            assert trace['obs'].shape == (2000, 10, 42)
            initial = trace['obs'][::200].reshape(100, 42).copy()
            initial_actions[condition['id']] = trace['action'][::200].reshape(100, 4).copy()
        if reference is None:
            assert condition['id'] == 'act-teacher-k30-first'
            reference = initial
        delta = float(np.max(np.abs(reference - initial)))
        assert delta == 0, (condition['id'], delta)
        row['initial_observation_max_difference'] = delta
        row['stage_counts'] = {
            name: sum(bool(r[name]) for r in records)
            for name in ['approached_once', 'is_grasped_once', 'lifted_once', 'is_obj_placed_once']
        }
        times = [r['first_grasp_step'] for r in records if r['first_grasp_step'] is not None]
        row['first_grasp_step_conditional_median'] = float(np.median(times)) if times else None
        row['first_grasp_step_denominator'] = len(times)
        assert sum(bool(r['success_once']) for r in records) == condition['success_once']
        assert sum(bool(r['success_at_end']) for r in records) == condition['success_at_end']
        sources.update({str(p): digest(p) for p in [episodes_path, trace_path]})
        rows.append(row)
    action_difference = np.abs(initial_actions['act-teacher-k30-first'] - initial_actions['act-c1-initial-mean'])
    paired = []
    for seed in [1, 2, 3, 4]:
        conditions = [x for x in summary['conditions'] if x['variant'] == 'U1' and x['training_seed'] == seed]
        bare = next(x for x in conditions if x['method'] == 'sac')
        wm = next(x for x in conditions if x['method'] == 'sac_wm')
        paired.append(dict(training_seed=seed, bare_success_rate=bare['success_rate'],
                           no_il_wm_success_rate=wm['success_rate'],
                           success_rate_difference=wm['success_rate'] - bare['success_rate'],
                           mean_return_difference=wm['mean_return'] - bare['mean_return']))
    result = dict(
        scope='Post-freeze descriptive evidence only. No tuning or extra simulation. A training seed is the replication unit.',
        act_conditions=rows,
        matched_completed_act_conditions=sum(r['status'] == 'completed' for r in rows),
        c1_vs_first_initial_action_max_abs=float(action_difference.max()),
        c1_vs_first_initial_action_mse=float(np.mean(action_difference ** 2)),
        c1_vs_first_interpretation='Numerically close initial actions are not a bitwise trajectory guarantee; one extra success is not evidence of better initialization.',
        u1_no_il_wm_vs_bare_by_training_seed=paired,
        new_environment_interactions=0, new_training_updates=0)
    (args.output / 'result.json').write_text(json.dumps(result, indent=2) + '\n')
    sources[__file__] = digest(Path(__file__))
    (args.output / 'sources.json').write_text(json.dumps(sources, indent=2) + '\n')
    np.savez_compressed(args.output / 'matched_initial_actions.npz', initial_observations=reference, **initial_actions)
    print(json.dumps(dict(completed_act_conditions=result['matched_completed_act_conditions'],
                          maximum_initial_action_difference=result['c1_vs_first_initial_action_max_abs'])))


if __name__ == '__main__':
    main()
