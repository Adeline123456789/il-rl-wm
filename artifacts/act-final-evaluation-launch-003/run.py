"""Run the frozen final registry once, then summarize all completed/failed conditions."""
import json
import os
from pathlib import Path
import subprocess
import time

root = Path(__file__).resolve().parents[2]
out = Path(__file__).resolve().parent
python = str(root / '.venv/bin/python')
status = dict(pid=os.getpid(), phase='running', exit_code=None, started_at=time.time())
(out / 'status.json').write_text(json.dumps(status, indent=2) + '\n')
try:
    assert json.loads((root / 'configs/act_final_test_gate.json').read_text())['released']
    assert json.loads((root / 'artifacts/act-u2-comparison-audit-001/exit.json').read_text())['exit_code'] == 0
    result = subprocess.run([python, 'scripts/run_final_evaluation.py',
        '--plan', 'configs/act_final_evaluation_plan.json',
        '--output', 'artifacts/act-final-evaluation-batch-003', *json.loads((out / 'resume.json').read_text())['arguments']], cwd=root)
    status['evaluation_exit_code'] = result.returncode
    summary = subprocess.run([python, 'scripts/summarize_final_results.py',
        '--plan', 'configs/act_final_evaluation_plan.json',
        '--output', 'artifacts/act-final-summary-003'], cwd=root,
        env=dict(os.environ, CUDA_VISIBLE_DEVICES=''))
    status['summary_exit_code'] = summary.returncode
    status.update(phase='completed' if result.returncode == summary.returncode == 0 else 'failed',
                  exit_code=0 if result.returncode == summary.returncode == 0 else 1)
except BaseException as error:
    status.update(phase='failed', exit_code=1, error=repr(error))
    raise
finally:
    status['ended_at'] = time.time()
    (out / 'status.json').write_text(json.dumps(status, indent=2) + '\n')
raise SystemExit(status['exit_code'])
