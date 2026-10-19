import os,time,subprocess,json
from pathlib import Path
p=Path('artifacts/act-final-review-cpu-001');start=time.time()
s=dict(pid=os.getpid(),phase='running',exit_code=None);(p/'status.json').write_text(json.dumps(s))
r=subprocess.run(['.venv/bin/python','scripts/review_act_final_evidence.py','--output',str(p/'analysis')],env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',PYTHONPATH=''))
s.update(phase='completed' if r.returncode==0 else 'failed',exit_code=r.returncode,elapsed_seconds=time.time()-start);(p/'status.json').write_text(json.dumps(s,indent=2));print(s)
