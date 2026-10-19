import os,time,json,subprocess
from pathlib import Path
p=Path('artifacts/act-final-delivery-audit-001');t=time.time()
status=dict(pid=os.getpid(),phase='waiting',exit_code=None,depends_on='artifacts/act-final-evaluation-launch-003/status.json')
(p/'status.json').write_text(json.dumps(status))
try:
 while True:
  upstream=json.loads(Path(status['depends_on']).read_text())
  if upstream['phase'] in ['completed','failed'] and upstream['exit_code'] is not None:break
  os.kill(upstream['pid'],0)
  if time.time()-t>7200:raise TimeoutError('Final audit wait expired; inspect original handle, never restart GPU')
  time.sleep(30)
 status['phase']='running';(p/'status.json').write_text(json.dumps(status))
 r=subprocess.run(['.venv/bin/python','scripts/audit_act_final_delivery.py','--output',str(p/'audit')],env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',PYTHONPATH=''))
 status.update(phase='completed' if r.returncode==0 else 'failed',exit_code=r.returncode,upstream_exit_code=upstream['exit_code'])
except BaseException as e:status.update(phase='failed',exit_code=1,error=repr(e))
finally:
 status['elapsed_seconds']=time.time()-t;(p/'status.json').write_text(json.dumps(status,indent=2));print(status)
