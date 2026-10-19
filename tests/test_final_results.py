"""Synthetic evidence only: test grouping and refusal of mixed/incomplete final tests."""
import copy
import json

import pytest

from scripts.summarize_final_results import summarize


def write(path, value):
    path.write_text(json.dumps(value))


def fixture(root):
    entries = [dict(id=f'{variant}-{seed}', framework='main', task='PickCube-v1', training_seed=seed,
                    variant=variant, training_interactions=524288,
                    training_role='development' if seed == 0 else 'independent_repeat', methods=['sac_wm'])
               for variant in ['U0', 'U1'] for seed in range(3)]
    entries += [dict(id=f'rlpd-{steps}', framework='rlpd', task='PickCube-v1', training_seed=0,
                     variant='RLPD', training_interactions=steps, training_role='development', methods=['RLPD'])
                for steps in [262144, 524288]]
    plan = dict(seed=20000, episodes_per_model_method=100, num_envs=10, entries=entries)
    for entry in entries:
        folder = root / f"e7-test-{entry['id']}-001"; folder.mkdir()
        write(folder / 'manifest.json', dict(ended_at='synthetic', exit_code=0, stop_reason=None))
        success = 20 * (entry['training_seed'] + 1) + (10 if entry['variant'] == 'U1' else 0)
        method = entry['methods'][0]
        episodes = [dict(method=method, episode=i, step=262144, episode_len=200, eval_seed=20000+i//10*10,
                         success_once=int(i < success), success_at_end=int(i < success),
                         **{'return': 100.0 if i < success else 0.0}) for i in range(100)]
        (folder / 'episodes.jsonl').write_text('\n'.join(json.dumps(ep) for ep in episodes))
        metrics = dict(success_once=success, success_at_end=success, mean_return=float(success), interactions=20000)
        result = dict(purpose='final_test', seed=20000, num_envs=10, new_training_interactions=0,
                      protocol=dict(env_id=entry['task'], max_episode_steps=200,reset_protocol='explicit_batch_seed'), training_seed=entry['training_seed'])
        if entry['framework'] == 'main':
            result.update(methods={method: metrics}, lineage_training_interactions=entry['training_interactions'])
        else:
            result.update(metrics, training_interactions=entry['training_interactions'])
        write(folder / 'result.json', result)
        write(folder / 'interactions.json', dict(phases=dict(final_test=dict(attempted_transitions=20000,
              completed_transitions=20000)), total_completed=20000, total_attempted=20000))
    return plan


def test_training_runs_roles_and_budgets_stay_separate(tmp_path):
    result = summarize(fixture(tmp_path), tmp_path)
    assert result['completed_conditions'] == 8
    groups = result['aggregates']
    independent = next(g for g in groups if g['variant'] == 'U0' and g['training_role'] == 'independent_repeat')
    assert independent['n_training_runs'] == 2 and independent['training_seeds'] == [1, 2]
    assert independent['mean_success_rate'] == .5
    assert independent['sample_sd_success_rate'] == pytest.approx(2 ** .5 * .1)
    assert independent['mean_return'] == 50 and independent['sample_sd_return'] == pytest.approx(2 ** .5 * 10)
    assert len([g for g in groups if g['variant'] == 'RLPD']) == 2
    assert len(result['paired_differences']) == 3
    assert all(p['success_rate_difference'] == pytest.approx(.1) for p in result['paired_differences'])


def test_failed_and_missing_runs_are_not_zero_scores(tmp_path):
    plan = fixture(tmp_path)
    write(tmp_path / 'e7-test-U0-0-001/manifest.json', dict(ended_at='synthetic', exit_code=1, stop_reason='timeout'))
    (tmp_path / 'e7-test-U1-0-001/manifest.json').unlink()
    result = summarize(plan, tmp_path)
    rows = {r['id']: r for r in result['conditions']}
    assert rows['U0-0']['status'] == 'failed' and rows['U0-0']['success_rate'] is None
    assert rows['U1-0']['status'] == 'not_started' and rows['U1-0']['success_rate'] is None
    assert len(result['paired_differences']) == 2


@pytest.mark.parametrize('fault', ['seed', 'budget', 'validation', 'duplicate_episode', 'partial_episode', 'reset_seed'])
def test_reject_mislabelled_or_incomplete_evidence(tmp_path, fault):
    plan = fixture(tmp_path); folder = tmp_path / 'e7-test-U0-0-001'
    if fault in ['seed', 'budget', 'validation']:
        result = json.loads((folder / 'result.json').read_text())
        result.update({'training_seed': 99} if fault == 'seed' else
                      {'lineage_training_interactions': 262144} if fault == 'budget' else {'purpose': 'validation'})
        write(folder / 'result.json', result)
    else:
        episodes = [json.loads(line) for line in (folder / 'episodes.jsonl').read_text().splitlines()]
        if fault == 'duplicate_episode': episodes[-1] = copy.deepcopy(episodes[0])
        elif fault == 'reset_seed': episodes[-1]['eval_seed'] = 20000
        else: episodes[-1]['episode_len'] = 199
        (folder / 'episodes.jsonl').write_text('\n'.join(json.dumps(ep) for ep in episodes))
    with pytest.raises(AssertionError): summarize(plan, tmp_path)
