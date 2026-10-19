"""Recompute registered final-test tables from whole-episode evidence, without ML imports."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics


IDENTITY = ['task', 'training_seed', 'variant', 'training_interactions', 'training_role']
GROUP = ['task', 'variant', 'training_interactions', 'training_role', 'method']


def read(path):
    return json.loads(path.read_text())


def summarize(plan, root):
    assert (plan['seed'], plan['episodes_per_model_method'], plan['num_envs']) == (20000, 100, 10)
    conditions, evidence = [], []
    for entry in plan['entries']:
        folder = root / f"e7-test-{entry['id']}-001"
        rows = [dict(id=entry['id'], **{key: entry[key] for key in IDENTITY}, method=method,
                     status='not_started', episodes=None, success_once=None, success_at_end=None,
                     success_rate=None, mean_return=None, evaluation_interactions=None)
                for method in entry['methods']]
        conditions.extend(rows)
        if not (folder / 'manifest.json').exists():
            continue
        manifest = read(folder / 'manifest.json')
        status = 'running' if manifest['ended_at'] is None else 'failed'
        if manifest['ended_at'] and manifest['exit_code'] == 0 and manifest['stop_reason'] is None:
            status = 'completed'
        for row in rows:
            row['status'] = status
        if status != 'completed':
            continue
        assert not (folder / 'INVALIDATED.json').exists(), entry['id']
        result = read(folder / 'result.json')
        counter = read(folder / 'interactions.json')
        episodes = [json.loads(line) for line in (folder / 'episodes.jsonl').read_text().splitlines()]
        assert result['purpose'] == 'final_test' and result['seed'] == 20000
        assert result['num_envs'] == 10 and result['new_training_interactions'] == 0
        assert result['protocol']['env_id'] == entry['task']
        assert result['protocol']['reset_protocol'] == 'explicit_batch_seed'
        assert result['training_seed'] == entry['training_seed']
        budget_key = 'lineage_training_interactions' if entry['framework'] == 'main' else 'training_interactions'
        assert result[budget_key] == entry['training_interactions']
        methods = result['methods'] if entry['framework'] == 'main' else {'RLPD': result}
        assert set(methods) == set(entry['methods']) == {ep['method'] for ep in episodes}
        assert len(episodes) == 100 * len(rows)
        total = 0
        for row in rows:
            selected = [ep for ep in episodes if ep['method'] == row['method']]
            assert len(selected) == 100 and {ep['episode'] for ep in selected} == set(range(100))
            assert all(ep['eval_seed'] == 20000 + ep['episode'] // 10 * 10 for ep in selected)
            assert len({ep['step'] for ep in selected}) == 1
            assert all(ep['episode_len'] == result['protocol']['max_episode_steps'] for ep in selected)
            assert all(ep['success_once'] in [0, 1] and ep['success_at_end'] in [0, 1]
                       and math.isfinite(ep['return']) for ep in selected)
            success = sum(ep['success_once'] for ep in selected)
            final_success = sum(ep['success_at_end'] for ep in selected)
            mean_return = statistics.mean(ep['return'] for ep in selected)
            interactions = int(sum(ep['episode_len'] for ep in selected))
            recorded = methods[row['method']]
            assert (recorded['success_once'], recorded['success_at_end'], recorded['interactions']) == (
                success, final_success, interactions)
            assert math.isclose(recorded['mean_return'], mean_return, rel_tol=1e-9, abs_tol=1e-9)
            row.update(episodes=100, success_once=int(success), success_at_end=int(final_success),
                       success_rate=success / 100, mean_return=mean_return, evaluation_interactions=interactions)
            total += interactions
        assert set(counter['phases']) == {'final_test'}
        phase = counter['phases']['final_test']
        assert phase['attempted_transitions'] == phase['completed_transitions'] == total
        assert counter['total_attempted'] == counter['total_completed'] == total
        evidence.append(dict(id=entry['id'], files={name: hashlib.sha256((folder / name).read_bytes()).hexdigest()
            for name in ['manifest.json', 'result.json', 'interactions.json', 'episodes.jsonl']}))
    complete = [row for row in conditions if row['status'] == 'completed']
    groups = {}
    for row in complete:
        groups.setdefault(tuple(row[key] for key in GROUP), []).append(row)
    aggregates = []
    for key, rows in groups.items():
        assert len({row['training_seed'] for row in rows}) == len(rows), 'Duplicate training run in aggregate'
        rates = [row['success_rate'] for row in rows]
        returns = [row['mean_return'] for row in rows]
        aggregates.append(dict(zip(GROUP, key), training_seeds=[row['training_seed'] for row in rows],
            n_training_runs=len(rows), mean_success_rate=statistics.mean(rates),
            sample_sd_success_rate=statistics.stdev(rates) if len(rates) > 1 else None,
            mean_return=statistics.mean(returns), sample_sd_return=statistics.stdev(returns) if len(returns) > 1 else None))
    paired = []
    index = {(r['task'], r['training_seed'], r['training_interactions'], r['method'], r['variant']): r for r in complete}
    for treatment, control in [('U1', 'U0'), ('C-LN', 'C'), ('C-LN continuation', 'C continuation')]:
        for row in complete:
            if row['variant'] != treatment:
                continue
            other = index.get((row['task'], row['training_seed'], row['training_interactions'], row['method'], control))
            if other is not None:
                paired.append(dict(task=row['task'], training_seed=row['training_seed'], training_role=row['training_role'],
                    training_interactions=row['training_interactions'], method=row['method'], treatment=treatment, control=control,
                    success_rate_difference=row['success_rate'] - other['success_rate'],
                    mean_return_difference=row['mean_return'] - other['mean_return']))
    return dict(conditions=conditions, aggregates=aggregates, paired_differences=paired, evidence=evidence,
        completed_conditions=len(complete), registered_conditions=len(conditions),
        definition='One row per registered model/method. Missing and failed runs have null scores. Aggregation uses training runs, separates development from independent repeats, tasks and interaction budgets. No significance or convergence claim.')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--plan', type=Path, default=Path('configs/final_evaluation_plan.json'))
    parser.add_argument('--root', type=Path, default=Path('artifacts'))
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = summarize(read(args.plan), args.root)
    result['plan_sha256'] = hashlib.sha256(args.plan.read_bytes()).hexdigest()
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / 'summary.json').write_text(json.dumps(result, indent=2) + '\n')
    with (args.output / 'conditions.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(result['conditions'][0]), lineterminator='\n')
        writer.writeheader(); writer.writerows(result['conditions'])


if __name__ == '__main__':
    main()
