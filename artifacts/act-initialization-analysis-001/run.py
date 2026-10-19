"""Read-only, bounded CPU summaries after each of the three authorized controls."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
os.chdir(ROOT)
OUT = Path('artifacts/act-initialization-analysis-001')
PLAN = Path('configs/act_initialization_execution_plan_002.json')
plan = json.loads(PLAN.read_text())
state = dict(pid=os.getpid(), phase='waiting', started_at=time.time(), exit_code=None,
             completed=[], new_environment_interactions=0, gpu_jobs_started=0)
def save():
    (OUT / 'status.json').write_text(json.dumps(state, indent=2) + '\n')
try:
    save()
    deadline = time.monotonic() + 14400
    for index, name in enumerate(plan['training_jobs'], 1):
        state.update(phase='waiting', current=name); save()
        while True:
            status = json.loads(Path('artifacts/act-initialization-batch-002/status.json').read_text())
            assert status['phase'] != 'failed', status
            manifest_path = Path('artifacts') / name / 'manifest.json'
            if manifest_path.exists():
                manifest = json.loads(manifest_path.read_text())
                if manifest['ended_at'] is not None:
                    assert manifest['exit_code'] == 0 and manifest['stop_reason'] is None, manifest
                    if (Path(plan['exports'][name]) / 'manifest.json').exists():
                        break
            assert time.monotonic() < deadline, 'Bounded CPU postprocessing wait expired'
            time.sleep(30)
        state['phase'] = 'summarizing'; save()
        command = [sys.executable, 'scripts/summarize_act_online.py', '--plan',
                   'configs/act_initialization_analysis_plan.json', '--output',
                   str(OUT / f'summary-{index:03}')]
        (OUT / f'command-{index:03}.json').write_text(json.dumps(command, indent=2) + '\n')
        subprocess.run(command, check=True, env=dict(os.environ, CUDA_VISIBLE_DEVICES=''))
        state['completed'].append(name); save()
    state.update(phase='completed', exit_code=0, current=None)
except BaseException as error:
    state.update(phase='failed', exit_code=1, error=repr(error))
    raise
finally:
    state['ended_at'] = time.time(); save()
