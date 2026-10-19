import json,os,subprocess,time
from pathlib import Path
out=Path('artifacts/act-r0-interface-cpu-001');s=dict(pid=os.getpid(),phase='running',started_at=time.time(),exit_code=None)
(out/'status.json').write_text(json.dumps(s,indent=2)+'\n')
try:
 subprocess.run(['.venv/bin/python','-m','pytest','-q','tests/test_sac_expert_initialization_controls.py','tests/test_sac_actor_initialization.py','tests/test_expert_replay.py','tests/test_sac_contract.py'],check=True)
 subprocess.run(['.venv/bin/python',str(out/'audit.py')],check=True)
 s.update(phase='completed',exit_code=0)
except BaseException as e:
 s.update(phase='failed',exit_code=1,error=repr(e));raise
finally:
 s['ended_at']=time.time();(out/'status.json').write_text(json.dumps(s,indent=2)+'\n')
