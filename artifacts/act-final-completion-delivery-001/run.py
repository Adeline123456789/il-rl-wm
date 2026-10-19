import json,os,subprocess,time
from pathlib import Path
p=Path('artifacts/act-final-completion-delivery-001');s=dict(pid=os.getpid(),started_at=time.time(),phase='running',exit_code=None)
try:
 r=subprocess.run(['.venv/bin/python','artifacts/act-final-completion-postprocess-001/update_delivery.py'],check=True)
 s.update(phase='completed',exit_code=0)
except BaseException as e:
 s.update(phase='failed',exit_code=1,error=repr(e));raise
finally:
 s['ended_at']=time.time();(p/'status.json').write_text(json.dumps(s,indent=2)+'\n')
