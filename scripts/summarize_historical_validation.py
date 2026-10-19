"""Recompute fixed-checkpoint historical validation, without reading final-test data."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics


def read(path):
    return json.loads(path.read_text())


def lines(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def summarize(plan, root):
    conditions, evidence, runs = [], {}, {}
    for entry in plan['entries']:
        name = entry['training_experiment']
        folder = root / name
        manifest = read(folder / 'manifest.json')
        assert manifest['ended_at'] and manifest['exit_code'] == 0 and manifest['stop_reason'] is None, name
        assert not (folder / 'INVALIDATED.json').exists(), name
        result = read(folder / 'result.json')
        counter = read(folder / 'interactions.json')
        assert not result['final_test_evaluated']
        config_name = 'sac_config.json' if entry['framework'] == 'main' else 'config.json'
        config = read(folder / config_name)
        assert config['env_id'] == entry['task'] and config['seed'] == entry['training_seed']
        horizon, control = {'PickCube-v1': (200, 'pd_ee_delta_pos'),
                            'StackCube-v1': (300, 'pd_ee_delta_pos'),
                            'PegInsertionSide-v1': (400, 'pd_ee_delta_pose')}[entry['task']]
        assert config['max_episode_steps'] == horizon and config['control_mode'] == control
        seed_key = 'eval_seed' if entry['framework'] == 'main' else 'validation_seed'
        assert config[seed_key] == 10000
        rows = lines(folder / 'validation_episodes.jsonl')
        assert set(counter['phases']) == {'training', 'validation'}
        for phase in counter['phases'].values():
            assert phase['attempted_transitions'] == phase['completed_transitions']
        assert sum(row['episode_len'] for row in rows) == result['validation_interactions'] == counter['phases']['validation']['completed_transitions']
        assert result['training_interactions'] == config['total_timesteps'] == counter['phases']['training']['completed_transitions']
        inherited = result.get('inherited_training_interactions', 0)
        step = entry['training_interactions'] - inherited
        if entry['framework'] == 'main':
            assert step == result['training_interactions']
            assert config['num_eval_steps'] % horizon == 0
            expected_count = config['num_eval_envs'] * config['num_eval_steps'] // horizon
        else:
            assert step in (262144, 524288)
            expected_count = config['validation_episodes']
        checkpoints = lines(folder / 'checkpoints.jsonl')
        location_key = 'path' if entry['framework'] == 'main' else 'directory'
        checkpoint = [c for c in checkpoints if c[location_key] == entry['checkpoint']]
        assert len(checkpoint) == 1 and checkpoint[0]['step'] == step, entry['id']
        checkpoint_hash = (checkpoint[0]['sha256'] if entry['framework'] == 'main'
                           else checkpoint[0]['files']['learner.msgpack']['sha256'])
        # Historical bare-SAC and isolated RLPD rows omit the method field.
        assert all('method' in r for r in rows) or len(entry['methods']) == 1
        for method in entry['methods']:
            selected = [r for r in rows if r['step'] == step and r.get('method', entry['methods'][0]) == method]
            assert len(selected) == expected_count, (entry['id'], method, len(selected), expected_count)
            assert all(r['episode_len'] == horizon and r['success_once'] in (0, 1)
                       and r['success_at_end'] in (0, 1) and math.isfinite(r['return']) for r in selected)
            if entry['framework'] == 'main':
                assert {r['episode'] for r in selected} == set(range(expected_count))
            else:
                assert {r['env'] for r in selected} == set(range(expected_count))
            conditions.append(dict(id=entry['id'], experiment=name, task=entry['task'],
                training_seed=entry['training_seed'], training_role=entry['training_role'],
                variant=entry['variant'], method=method, training_interactions=entry['training_interactions'],
                episode_horizon=horizon, control_mode=control,
                checkpoint_step=step, checkpoint_sha256=checkpoint_hash,
                episodes=expected_count, success_once=int(sum(r['success_once'] for r in selected)),
                success_at_end=int(sum(r['success_at_end'] for r in selected)),
                mean_return=statistics.mean(r['return'] for r in selected)))
        runs[name] = dict(gpu_job_seconds=manifest['elapsed_seconds'],
            code_commit=manifest['code_commit'], training_interactions=result['training_interactions'],
            validation_interactions=result['validation_interactions'])
        evidence[name] = {f: hashlib.sha256((folder / f).read_bytes()).hexdigest()
            for f in ['manifest.json', 'result.json', config_name, 'interactions.json',
                      'validation_episodes.jsonl', 'checkpoints.jsonl']}
    groups = {}
    for row in conditions:
        key = (row['task'], row['variant'], row['method'], row['training_interactions'], row['training_role'])
        groups.setdefault(key, []).append(row)
    aggregates = []
    for key, rows in groups.items():
        assert len({r['training_seed'] for r in rows}) == len(rows)
        rates = [r['success_once'] / r['episodes'] for r in rows]
        aggregates.append(dict(zip(['task', 'variant', 'method', 'training_interactions', 'training_role'], key),
            seeds=[r['training_seed'] for r in rows], n_training_runs=len(rows),
            mean_success_rate=statistics.mean(rates),
            sample_sd_success_rate=statistics.stdev(rates) if len(rates) > 1 else None))
    return dict(conditions=conditions, aggregates=aggregates, runs=runs, evidence=evidence,
        final_test_accessed=False, new_environment_interactions=0,
        scope='Existing validation only; fixed registered checkpoints, not new final-test evidence. '
              'Development seed0 is separate from independent training seeds. Legacy reset protocol '
              'does not establish identical initial states across stochastic planning methods. '
              'RLPD checkpoints share one training run and are not equal-compute controls.')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--plan', type=Path, default=Path('configs/final_evaluation_plan.json'))
    parser.add_argument('--root', type=Path, default=Path('artifacts'))
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = summarize(read(args.plan), args.root)
    result['plan_sha256'] = hashlib.sha256(args.plan.read_bytes()).hexdigest()
    args.output.mkdir(exist_ok=False, parents=True)
    (args.output / 'summary.json').write_text(json.dumps(result, indent=2) + '\n')
    with (args.output / 'conditions.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(result['conditions'][0]), lineterminator='\n')
        writer.writeheader(); writer.writerows(result['conditions'])
    print(json.dumps(dict(conditions=len(result['conditions']), runs=len(result['runs']))))


if __name__ == '__main__':
    main()
