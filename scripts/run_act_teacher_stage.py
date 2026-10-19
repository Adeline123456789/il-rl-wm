"""Bounded ACT-T0 smoke and K=8/16/30 comparison after existing training."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def main():
    p = argparse.ArgumentParser(); p.add_argument('--output', type=Path, required=True)
    args = p.parse_args(); args.output.mkdir(parents=True, exist_ok=False)
    record = dict(phase='running', completed=[], current=None, stage='ACT-T0')
    def save():
        tmp = args.output/'status.tmp'; tmp.write_text(json.dumps(record, indent=2)+'\n')
        tmp.replace(args.output/'status.json')
    try:
        for check in ['act-cpu-tests-004', 'act-real-data-cpu-003']:
            assert json.loads((ROOT/'artifacts'/check/'status.json').read_text())['exit_code'] == 0
        budget = json.loads((ROOT/'research_state.json').read_text())['budget']
        manifests = [json.loads(p.read_text()) for p in (ROOT/'artifacts').glob('*/manifest.json')]
        assert all(m['ended_at'] for m in manifests), 'Wait for the active GPU job'
        used = sum(m['elapsed_seconds'] for m in manifests)
        available = budget['gpu_hours_limit']*3600-used-budget['final_evaluation_reserve_seconds']
        record.update(start_budget_used_seconds=used, available_training_seconds=available,
                      stage_limit_seconds=5700, final_test_reserve_seconds=budget['final_evaluation_reserve_seconds'])
        assert available >= 2100, 'Insufficient existing budget for bounded ACT-T0; no new GPU work launched'
        for name, config, seconds in [('act-t0-smoke-001', 'act_t0_smoke.json', 300),
                                     ('act-t0-seed0-001', 'act_t0_seed0.json', 1800),
                                     ('act-t0-k16-seed0-001', 'act_t0_k16_seed0.json', 1800),
                                     ('act-t0-k30-seed0-001', 'act_t0_k30_seed0.json', 1800)]:
            current_budget = json.loads((ROOT/'research_state.json').read_text())['budget']
            available = current_budget['gpu_hours_limit']*3600-current_budget['gpu_job_seconds_used']-current_budget['final_evaluation_reserve_seconds']
            assert available >= seconds, 'Remaining budget cannot cover next registered job; preserve unrun condition'
            if name == 'act-t0-seed0-001':
                smoke = ROOT/'artifacts/act-t0-smoke-001'
                smoke_config = json.loads((smoke/'config.json').read_text())
                full_config = json.loads((ROOT/'configs'/config).read_text())
                assert smoke_config['batch_size'] == full_config['batch_size']
                assert smoke_config['model'] == full_config['model']
                last = json.loads((smoke/'train.jsonl').read_text().splitlines()[-1])
                duration = last['elapsed_seconds']/last['step']*10000
                decision = dict(training_updates=10000, measured_smoke_seconds_per_update=last['elapsed_seconds']/last['step'],
                    projected_training_seconds=duration, budget_seconds=1800,
                    rule='Require 1.5x observed K8 training cost + 300s validation/video allowance to fit; no hidden shortening')
                (args.output/'resource_decision.json').write_text(json.dumps(decision, indent=2))
                assert 1.5*duration+300 <= 1800, 'Measured throughput does not support registered length; stop for bounded redesign'
            record['current'] = name; save()
            subprocess.run([sys.executable, 'scripts/run_experiment.py', '--id', name,
                '--seconds', str(seconds), '--', sys.executable, 'scripts/train_act_teacher.py',
                '--config', 'configs/'+config, '--id', name, '--output', 'artifacts/'+name],
                cwd=ROOT, check=True)
            record['completed'].append(name)
        results = {name: json.loads((ROOT/'artifacts'/name/'result.json').read_text())
                   for name in record['completed'] if 'smoke' not in name}
        passed = any(r['teacher_gate_passed'] for r in results.values())
        record.update(phase='completed', current=None, teacher_gate_passed=passed,
            results={name: dict(first=r['final_validation'], ensemble=r['ensemble_validation']) for name,r in results.items()},
            next_stage='ACT-D1 expert-state-only distillation; freeze teacher using development validation' if passed
                else 'F-BC/ACT failure diagnosis; do not start distillation or WM')
    except BaseException as error:
        record.update(phase='failed', error=repr(error)); raise
    finally:
        save()


if __name__ == '__main__':
    main()
