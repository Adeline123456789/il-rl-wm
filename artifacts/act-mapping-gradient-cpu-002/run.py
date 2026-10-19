import subprocess,json,time,os
from pathlib import Path
p=Path('artifacts/act-mapping-gradient-cpu-002')
cmd=['.venv/bin/python', 'scripts/diagnose_act_mapping_gradients.py', '--output', 'artifacts/act-mapping-gradient-cpu-002/analysis']
t=time.time()
r=subprocess.run(cmd,env=dict(os.environ,CUDA_VISIBLE_DEVICES="",OMP_NUM_THREADS="2",PYTHONPATH=""))
(p/"status.json").write_text(json.dumps(dict(phase="completed" if r.returncode==0 else "failed",pid=os.getpid(),exit_code=r.returncode,elapsed_seconds=time.time()-t)))
