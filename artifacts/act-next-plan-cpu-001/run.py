import subprocess,json,os,time
from pathlib import Path
p=Path('artifacts/act-next-plan-cpu-001');t=time.time();r=subprocess.run(['.venv/bin/python','artifacts/act-next-plan-cpu-001/prepare.py'],env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2'))
(p/'status.json').write_text(json.dumps(dict(pid=os.getpid(),phase='completed' if r.returncode==0 else 'failed',exit_code=r.returncode,elapsed_seconds=time.time()-t)))
