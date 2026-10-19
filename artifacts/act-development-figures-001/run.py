import os,time,json,subprocess
from pathlib import Path
p=Path('artifacts/act-development-figures-001');t=time.time();qa=Path('/home/midea/.codex/skills/nature-figure/scripts')
commands=[['.venv/bin/python',str(qa/'validate_figure.py'),'scripts/plot_act_development_review.py'],['.venv/bin/python','scripts/plot_act_development_review.py','--output',str(p/'figures'),'--qa-tools',str(qa)]]
for name in ['online_success','gripper_mean_drift']:
 pdf=p/'figures'/f'{name}.pdf'
 commands.extend([['.venv/bin/python',str(qa/'audit_pdf_text.py'),str(pdf),'--min-pt','5','--json'],['.venv/bin/python',str(qa/'audit_figure_collisions.py'),str(pdf),'--json-out',str(p/'figures'/f'{name}.collision.json')]])
status=dict(pid=os.getpid(),phase='running',exit_code=None,commands=[]);(p/'status.json').write_text(json.dumps(status))
try:
 for i,cmd in enumerate(commands):
  with (p/f'command-{i}.log').open('w') as f:r=subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',PYTHONPATH=''))
  status['commands'].append(dict(command=cmd,exit_code=r.returncode))
  if r.returncode:raise RuntimeError(f'Command{i} failed with{r.returncode}')
 status.update(phase='completed',exit_code=0)
except BaseException as e:status.update(phase='failed',exit_code=1,error=repr(e))
finally:status['elapsed_seconds']=time.time()-t;(p/'status.json').write_text(json.dumps(status,indent=2));print(status)
