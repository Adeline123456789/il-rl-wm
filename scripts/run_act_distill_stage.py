import json,subprocess,sys,time,os
from pathlib import Path
out=Path('artifacts/act-d1-stage-001');out.mkdir()
status=dict(pid=os.getpid(),phase='running',completed=[],started_at=time.time())
def save(): (out/'status.json').write_text(json.dumps(status,indent=2)+'\n')
save()
try:
 assert json.loads(Path('artifacts/act-d1-tests-001/exit.json').read_text())['exit_code']==0
 for name,cfg,seconds in [('act-d1-smoke-001','act_d1_smoke.json',300),('act-d1-seed0-001','act_d1_seed0.json',1200)]:
  budget=json.loads(Path('research_state.json').read_text())['budget']
  assert 86400-budget['gpu_job_seconds_used']-7200>=seconds
  status['current']=name;save()
  subprocess.run([sys.executable,'scripts/run_experiment.py','--id',name,'--seconds',str(seconds),'--',sys.executable,'scripts/distill_act_sac.py','--config','configs/'+cfg,'--id',name,'--output','artifacts/'+name],check=True)
  status['completed'].append(name)
 status.update(phase='completed',exit_code=0)
except BaseException as exc:
 status.update(phase='failed',exit_code=1,error=repr(exc))
finally:
 status['ended_at']=time.time();save()
