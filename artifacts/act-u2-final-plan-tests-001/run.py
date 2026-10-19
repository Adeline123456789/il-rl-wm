import subprocess,os,json
from pathlib import Path
r=subprocess.run(['.venv/bin/python','-m','pytest','-q','tests/test_act_final_evidence.py'],env=dict(os.environ,PYTEST_DISABLE_PLUGIN_AUTOLOAD='1',PYTHONPATH='',CUDA_VISIBLE_DEVICES=''))
Path('artifacts/act-u2-final-plan-tests-001/exit.json').write_text(json.dumps(dict(pid=os.getpid(),exit_code=r.returncode)))
raise SystemExit(r.returncode)
