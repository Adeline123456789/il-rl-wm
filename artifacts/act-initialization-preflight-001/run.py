import json,os,subprocess,time
from pathlib import Path
out=Path(__file__).resolve().parent
s=dict(pid=os.getpid(),phase='running',started_at=time.time(),exit_code=None)
(out/'status.json').write_text(json.dumps(s,indent=2)+'\n')
try:
 subprocess.run(['.venv/bin/python','-m','pytest','-q','tests/test_additional_budget.py','tests/test_act_inference.py','tests/test_act_final_evidence.py','tests/test_sac_expert_initialization_controls.py','tests/test_sac_actor_initialization.py','tests/test_expert_replay.py','tests/test_sac_contract.py'],check=True)
 s.update(phase='completed',exit_code=0)
except BaseException as error:
 s.update(phase='failed',exit_code=1,error=repr(error));raise
finally:
 s['ended_at']=time.time();(out/'status.json').write_text(json.dumps(s,indent=2)+'\n')
