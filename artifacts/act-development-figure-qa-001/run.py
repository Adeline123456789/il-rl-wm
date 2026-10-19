import os,time,json,subprocess,sys
from pathlib import Path
p=Path('artifacts/act-development-figure-qa-001');t=time.time();qa=Path('/home/midea/.codex/skills/nature-figure/scripts');figs=Path('artifacts/act-development-figures-002/figures')
commands=[]
for name in ['online_success','gripper_mean_drift']:
 commands.extend([[sys.executable,str(qa/'audit_pdf_text.py'),str(figs/f'{name}.pdf'),'--min-pt','5','--json'],[sys.executable,str(qa/'audit_figure_collisions.py'),str(figs/f'{name}.pdf'),'--json-out',str(p/f'{name}.collision.json')]])
records=[]
for i,cmd in enumerate(commands):
 with (p/f'command-{i}.log').open('w') as f:r=subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT)
 records.append(dict(command=cmd,exit_code=r.returncode))
status=dict(pid=os.getpid(),phase='completed' if all(r['exit_code']==0 for r in records) else 'failed',exit_code=0 if all(r['exit_code']==0 for r in records) else 1,commands=records,elapsed_seconds=time.time()-t)
(p/'status.json').write_text(json.dumps(status,indent=2));print(status)
