import subprocess,os,json
from pathlib import Path
out=Path('artifacts/act-u2-cpu-tests-002')
command=['.venv/bin/python','-m','pytest','-q','tests/test_act_retention.py','tests/test_act_parameter_transfer.py','tests/test_act_online_drift.py']
r=subprocess.run(command,env=dict(os.environ,PYTEST_DISABLE_PLUGIN_AUTOLOAD='1',PYTHONPATH='',CUDA_VISIBLE_DEVICES=''))
(out/'exit.json').write_text(json.dumps(dict(pid=os.getpid(),command=command,exit_code=r.returncode)))
raise SystemExit(r.returncode)
