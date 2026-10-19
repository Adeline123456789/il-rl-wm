import json,subprocess,os
from pathlib import Path
out=Path('artifacts/act-u2-postprocess-launch-001')
status=dict(pid=os.getpid(),phase='running',completed=[],exit_code=None,GPU_use=False)
try:
 for cmd in json.loads((out/'commands.json').read_text()):
  subprocess.run(cmd,check=True,env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2'))
  status['completed'].append(cmd)
 status.update(phase='completed',exit_code=0)
except BaseException as e:
 status.update(phase='failed',exit_code=1,error=repr(e));raise
finally: (out/'status.json').write_text(json.dumps(status,indent=2)+'\n')
