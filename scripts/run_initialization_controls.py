"""One frozen R0/R1/R3 batch: resource gates, serial training, fixed final tests."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.run_experiment import write_json


def read(path):
    return json.loads(Path(path).read_text())


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    os.chdir(ROOT)
    parser = argparse.ArgumentParser()
    parser.add_argument('--plan', type=Path, default=Path('configs/act_initialization_execution_plan.json'))
    parser.add_argument('--output', type=Path, default=Path('artifacts/act-initialization-batch-001'))
    args = parser.parse_args()
    plan_path = args.plan
    plan = read(plan_path)
    auth_path = Path(plan['authorization'])
    authorization = read(auth_path)
    assert authorization['status'] == 'authorized' and authorization['total_seconds'] == 11500
    assert read(plan.get('preflight', 'artifacts/act-initialization-preflight-002/status.json'))['exit_code'] == 0
    out = args.output
    out.mkdir(exist_ok=False)
    state = dict(pid=os.getpid(), phase='running', completed=[], current=None,
                 started_at=time.time(), exit_code=None, plan_sha256=digest(plan_path),
                 authorization_sha256=digest(auth_path))
    def save():
        write_json(out / 'status.json', state)
    def run_job(name):
        assert digest(auth_path) == state['authorization_sha256']
        assert all(digest(path) == sha for path, sha in plan['runtime_sources'].items())
        state['current'] = name; save()
        job = authorization['jobs'][name]
        subprocess.run([sys.executable, 'scripts/run_experiment.py', '--id', name,
                        '--seconds', str(job['cap_seconds']), '--budget-file', str(auth_path),
                        '--', *job['command']], check=True)
        state['completed'].append(name); save()
    def check_training(name, smoke=False):
        folder = Path('artifacts') / name
        actual = read(folder / 'actor_initialization.json')
        assert [actual['critic1_hash'], actual['critic2_hash']] == plan['fixed_critic_hashes']
        config = read(folder / 'sac_config.json')
        for key, value in plan['expected_configs'][name].items():
            assert config[key] == value, (name, key, config.get(key), value)
        result = read(folder / 'result.json')
        assert result['training_interactions'] == (8192 if smoke else 262144)
        assert result['gradient_updates'] == (1040 if smoke else 64528)
        sampling = read(folder / 'sampling.json')
        assert sampling['expert_samples'] == sampling['online_samples'] == (532480 if smoke else 33038336)
        assert sampling['expert_fraction'] == .5
        return result
    save()
    try:
        projections = {}
        for name in plan['smoke_jobs']:
            if name in plan.get('completed_smoke_jobs', []):
                manifest = read(Path('artifacts') / name / 'manifest.json')
                assert manifest['ended_at'] and manifest['exit_code'] == 0 and manifest['stop_reason'] is None
                assert manifest['command'] == authorization['jobs'][name]['command']
                assert manifest['budget_id'] == authorization['budget_id']
                state.setdefault('reused_completed_smokes', []).append(name); save()
            else:
                run_job(name)
            result = check_training(name, smoke=True)
            folder = Path('artifacts') / name
            manifest = read(folder / 'manifest.json')
            rows = [json.loads(line) for line in (folder / 'metrics.jsonl').read_text().splitlines()]
            final = {r['metric']: r['value'] for r in rows if r['step'] == 8192}
            updates, rollout = final['time/total_update_time'], final['time/total_rollout_time']
            estimate_record = project_full_duration(manifest['elapsed_seconds'], result['elapsed_seconds'], updates, rollout)
            estimate = estimate_record['projected_full_seconds']
            cap = plan['smoke_full_cap'][name]
            projections[name] = dict(**estimate_record, measured_seconds=manifest['elapsed_seconds'], update_seconds=updates,
                rollout_seconds=rollout,
                projected_with_margin=estimate * 1.35 + 60, cap_seconds=cap,
                rule='One-time startup/cleanup once; scale in-loop residual 2 to17 evaluations, updates/rollouts;35% +60s. No score-based gate')
            write_json(out / 'resource_gate.json', projections)
            assert estimate * 1.35 + 60 <= cap, 'Measured resource projection exceeds registered cap; stop without changing budget'
        state['phase'] = 'training'; save()
        cpu_env = dict(os.environ, CUDA_VISIBLE_DEVICES='')
        for name in plan['training_jobs']:
            run_job(name); check_training(name)
            subprocess.run([sys.executable, 'scripts/export_act_online_actor.py',
                '--checkpoint', f'checkpoints/{name}/final.pt', '--output', plan['exports'][name]],
                env=cpu_env, check=True)
        evaluation = read(plan['evaluation_plan'])
        frozen = {}
        for entry in evaluation['entries']:
            checkpoint = entry['checkpoint']
            exported = read(Path(checkpoint).parent / 'manifest.json')
            assert digest(checkpoint) == exported['portable_sha256'] and exported['all_actor_tensors_equal']
            frozen[checkpoint] = digest(checkpoint)
        write_json(out / 'fixed_final_checkpoints.json', frozen)
        write_json(Path(plan['test_gate']), dict(released=True, released_at=time.time(),
            reason='All registered training finished; fixed final only; no score selection',
            evaluation_plan_sha256=digest(plan['evaluation_plan']), checkpoint_hashes=frozen))
        state['phase'] = 'final_evaluation'; save()
        for name in plan['evaluation_jobs']:
            assert all(digest(path) == sha for path, sha in frozen.items())
            run_job(name)
        subprocess.run([sys.executable, 'scripts/summarize_final_results.py',
            '--plan', plan['evaluation_plan'], '--output', 'artifacts/act-initialization-summary-001'],
            env=cpu_env, check=True)
        state.update(phase='completed', exit_code=0, current=None)
    except BaseException as error:
        state.update(phase='failed', exit_code=1, error=repr(error)); raise
    finally:
        state['ended_at'] = time.time(); save()


def project_full_duration(job_seconds, loop_seconds, update_seconds, rollout_seconds):
    """The training result timer starts after setup; do not repeat startup17/2 times."""
    assert 0 <= update_seconds + rollout_seconds <= loop_seconds <= job_seconds
    setup = job_seconds - loop_seconds
    recurring = loop_seconds - update_seconds - rollout_seconds
    projected = update_seconds / 1040 * 64528 + rollout_seconds / 8192 * 262144 + recurring / 2 * 17 + setup
    return dict(one_time_setup_cleanup_seconds=setup, in_loop_other_seconds=recurring,
                projected_full_seconds=projected,
                previous_all_overhead_scaled_seconds=projected + setup * (17 / 2 - 1))


if __name__ == '__main__':
    main()
