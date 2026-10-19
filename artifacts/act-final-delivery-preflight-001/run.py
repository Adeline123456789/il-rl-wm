import subprocess,os,json,time
from pathlib import Path
p=Path('artifacts/act-final-delivery-preflight-001');t=time.time();r=subprocess.run(['.venv/bin/python','scripts/audit_act_final_delivery.py','--allow-running-snapshot','--output',str(p/'audit')],env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',PYTHONPATH=''))
(p/'status.json').write_text(json.dumps(dict(pid=os.getpid(),phase='completed' if r.returncode==0 else 'failed',exit_code=r.returncode,elapsed_seconds=time.time()-t)))
