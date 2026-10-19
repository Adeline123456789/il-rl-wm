"""One user-authorized, 300-second U2 completion. No model selection or retries."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.run_final_evaluation import act_checkpoint_hash
from scripts.run_experiment import ensure_gpu_idle, write_json


def main():
    os.chdir(ROOT)
    out = ROOT / 'artifacts/act-final-completion-001'
    out.mkdir(exist_ok=False)
    started = time.monotonic()
    status = dict(pid=os.getpid(), phase='running', started_at=time.time(), completed=[], exit_code=None)
    def save():
        status['elapsed_seconds'] = time.monotonic() - started
        write_json(out / 'status.json', status)
    save()
    try:
        auth = json.loads(Path('configs/act_final_completion_authorization.json').read_text())
        assert auth['user_message'] == '批准补评 300 秒' and auth['additional_seconds'] == 300
        plan_path = Path('configs/act_final_evaluation_plan.json')
        assert hashlib.sha256(plan_path.read_bytes()).hexdigest() == auth['original_plan_sha256']
        plan = json.loads(plan_path.read_text())
        mapping = json.loads(Path(auth['attempt_map']).read_text())
        assert list(mapping) == ['act-u2-final-mean', 'act-u2-final-stochastic']
        assert json.loads(Path('configs/act_final_test_gate.json').read_text())['released']
        manifests = [json.loads(p.read_text()) for p in Path('artifacts').glob('*/manifest.json')]
        assert all(m['ended_at'] for m in manifests)
        used = sum(m['elapsed_seconds'] for m in manifests)
        assert 86400 - used >= 300
        status['gpu_seconds_before'] = used
        assert json.loads(Path('artifacts/act-u2-retention-seed0-001/manifest.json').read_text())['exit_code'] == 0
        entries = {e['id']: e for e in plan['entries']}
        for key, experiment in mapping.items():
            entry = entries[key]
            assert act_checkpoint_hash(entry, ROOT) == auth['checkpoint_sha256']
            assert not (Path('artifacts') / experiment).exists()
        for key, experiment in mapping.items():
            ensure_gpu_idle()
            entry = entries[key]
            status['current'] = key
            save()
            remaining = 300 - (time.monotonic() - started) - 30
            assert remaining > 0, 'Completion allowance exhausted; no retry'
            command = [str(ROOT / '.venv/bin/python'), 'scripts/run_experiment.py', '--id', experiment,
                '--seconds', str(min(120, remaining)), '--', str(ROOT / '.venv/bin/python'),
                'scripts/evaluate_act_checkpoint.py', '--checkpoint', entry['checkpoint'],
                '--kind', entry['kind'], '--mode', entry['mode'], '--training-seed', str(entry['training_seed']),
                '--training-interactions', str(entry['training_interactions']), '--output', f'artifacts/{experiment}',
                '--seed', '20000', '--episodes', '100', '--num-envs', '10']
            subprocess.run(command, check=True)
            status['completed'].append(dict(id=key, experiment=experiment))
            save()
        status.update(phase='completed', current=None, exit_code=0)
    except BaseException as error:
        status.update(phase='failed', exit_code=1, error=repr(error))
        raise
    finally:
        status['ended_at'] = time.time()
        save()
    return status['exit_code']


if __name__ == '__main__':
    raise SystemExit(main())
