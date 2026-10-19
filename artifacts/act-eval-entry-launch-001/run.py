import json,subprocess,sys
from pathlib import Path
state=json.loads(Path('research_state.json').read_text())
assert state['budget']['gpu_hours_limit']*3600-state['budget']['gpu_job_seconds_used']-state['budget']['final_evaluation_reserve_seconds']>=300
command=[sys.executable,'scripts/run_experiment.py','--id','act-independent-eval-smoke-001','--seconds','300','--',sys.executable,'scripts/check_act_evaluation_entry.py','--output','artifacts/act-independent-eval-smoke-001']
raise SystemExit(subprocess.call(command))
