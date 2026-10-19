"""Audit completed frozen T2--T6 runs; unfinished conditions keep null outcomes."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics


def read(path):
    return json.loads(path.read_text())


def rows(path):
    return [json.loads(x) for x in path.read_text().splitlines()]


def summarize(plan, root):
    conditions, curves, evidence = [], [], {}
    reference = None
    shared = ['seed', 'env_id', 'control_mode', 'max_episode_steps', 'total_timesteps',
              'num_envs', 'num_eval_envs', 'num_eval_steps', 'training_freq', 'utd',
              'batch_size', 'buffer_size', 'learning_starts', 'eval_freq', 'log_freq',
              'gamma', 'critic_layer_norm', 'initial_log_std', 'matched_eval_resets',
              'autotune', 'policy_lr', 'q_lr', 'alpha', 'tau', 'policy_frequency', 'bootstrap_at_done']
    for job in plan['jobs']:
        folder = root / job['id']
        row = dict(id=job['id'], status='not_started', initial_success=None, final_success=None,
                   episodes_per_checkpoint=40, training_interactions=None, gradient_updates=None,
                   gpu_job_seconds=None, peak_device_mib=None, actor_parameters=None)
        conditions.append(row)
        if not (folder / 'manifest.json').exists():
            continue
        manifest = read(folder / 'manifest.json')
        row['status'] = 'running' if manifest['ended_at'] is None else 'failed'
        if not manifest['ended_at'] or manifest['exit_code'] != 0 or manifest['stop_reason'] is not None:
            continue
        assert manifest['command'] == job['command']
        assert not (folder / 'INVALIDATED.json').exists()
        config = read(folder / 'sac_config.json')
        identity = {k: config[k] for k in shared}
        if reference is None:
            reference = identity
        assert identity == reference, 'Scientific controls differ'
        assert config['seed'] == 0 and config['eval_seed'] == 10000
        assert config['critic_layer_norm'] and config['matched_eval_resets']
        assert config['total_timesteps'] == 262144 and not config.get('wm_checkpoint')
        result = read(folder / 'result.json')
        counter = read(folder / 'interactions.json')
        assert result['training_interactions'] == 262144 and result['gradient_updates'] == 64528
        assert result['validation_interactions'] == 136000 and not result['final_test_evaluated']
        assert set(counter['phases']) == {'training', 'validation'}
        for phase, count in [('training', 262144), ('validation', 136000)]:
            assert counter['phases'][phase]['attempted_transitions'] == counter['phases'][phase]['completed_transitions'] == count
        initial = read(folder / 'actor_initialization.json')
        assert [initial['critic1_hash'], initial['critic2_hash']] == plan['fixed_critic_hashes']
        assert initial['variance_trainable'] and not initial['teacher_calls_in_sac_updates']
        episodes = rows(folder / 'validation_episodes.jsonl')
        assert len(episodes) == 680 and {e['step'] for e in episodes} == set(range(0, 262145, 16384))
        for step in range(0, 262145, 16384):
            batch = [e for e in episodes if e['step'] == step]
            assert len(batch) == 40 and {e['episode'] for e in batch} == set(range(40))
            assert all(e['eval_seed'] == 10000 + e['episode'] // 8 * 8 and e['episode_len'] == 200 for e in batch)
            assert all(e['success_once'] in (0, 1) and e['success_at_end'] in (0, 1) and math.isfinite(e['return']) for e in batch)
            curves.append(dict(id=job['id'], step=step, episodes=40,
                success_once=int(sum(e['success_once'] for e in batch)),
                success_at_end=int(sum(e['success_at_end'] for e in batch)),
                mean_return=statistics.mean(e['return'] for e in batch),
                grasped_once=int(sum(e['is_grasped_once'] for e in batch)),
                placed_once=int(sum(e['is_obj_placed_once'] for e in batch))))
        metrics = rows(folder / 'metrics.jsonl')
        assert all(math.isfinite(e['value']) for e in metrics)
        training = rows(folder / 'train_episodes.jsonl')
        resources = rows(folder / 'resources.jsonl')
        stage = [c for c in curves if c['id'] == job['id']]
        checkpoints = rows(folder / 'checkpoints.jsonl')
        final = [c for c in checkpoints if c['final']]
        assert len(final) == 1 and final[0]['step'] == 262144
        row.update(status='completed', initial_success=stage[0]['success_once'], final_success=stage[-1]['success_once'],
            training_interactions=262144, gradient_updates=64528, gpu_job_seconds=manifest['elapsed_seconds'],
            peak_device_mib=max(r['memory_mib'] for r in resources), actor_parameters=initial['actor_parameters'],
            checkpoint_sha256=final[0]['sha256'], training_complete_episodes=len(training),
            training_success_episodes=int(sum(e['success_once'] for e in training)),
            losses={key:dict(min=min(e['value'] for e in metrics if e['metric']==key),
                            max=max(e['value'] for e in metrics if e['metric']==key),
                            final=[e['value'] for e in metrics if e['metric']==key][-1])
                    for key in ['losses/alpha','losses/qf1_values','losses/qf2_values','losses/actor_loss']})
        evidence[job['id']] = {name:hashlib.sha256((folder/name).read_bytes()).hexdigest() for name in
            ['manifest.json','sac_config.json','actor_initialization.json','result.json','interactions.json',
             'validation_episodes.jsonl','train_episodes.jsonl','metrics.jsonl','resources.jsonl','checkpoints.jsonl']}
    return dict(conditions=conditions, curves=curves, evidence=evidence, shared_config=reference,
        scope='Seed0 development; fixed final, validation only. Curves are not independent training repeats. '
              'Initial teacher/transfer results have separate offline costs; no ACT-WM conclusion.',
        final_test_accessed=False, new_environment_interactions=0)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--plan',type=Path,default=Path('artifacts/act-online-budget-002/plan.json'))
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    result=summarize(read(args.plan),Path('artifacts'))
    result['plan_sha256']=hashlib.sha256(args.plan.read_bytes()).hexdigest()
    args.output.mkdir(parents=True,exist_ok=False)
    (args.output/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    if result['curves']:
        with (args.output/'curves.csv').open('w',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=list(result['curves'][0]),lineterminator='\n')
            writer.writeheader();writer.writerows(result['curves'])
    print(json.dumps(result['conditions'],indent=2))


if __name__=='__main__':main()
