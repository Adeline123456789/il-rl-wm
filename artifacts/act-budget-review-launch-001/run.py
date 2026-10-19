import subprocess,os,json,time
from pathlib import Path
p=Path('artifacts/act-budget-review-launch-001');t=time.time();r=subprocess.run(['.venv/bin/python',str(p/'analyze.py')],env=dict(os.environ,CUDA_VISIBLE_DEVICES=''))
(p/'status.json').write_text(json.dumps(dict(pid=os.getpid(),phase='completed' if r.returncode==0 else 'failed',exit_code=r.returncode,elapsed_seconds=time.time()-t)))
