import json,sys,tempfile
from pathlib import Path
sys.path.insert(0,str(Path.cwd()))
from scripts import run_pickcube_repeat as main
from scripts import run_u1_repeat as u1
for seed in [1,2]:
 assert main.plan(seed)==json.loads(Path(f'artifacts/repeat-plan-check-001/seed{seed}.json').read_text())
for seed in [0,1,2]:
 assert u1.plan(seed)==json.loads(Path(f'artifacts/u1-and-task-plan-check-002/u1-seed{seed}.json').read_text())
out=Path('artifacts/five-seed-plan-check-001')
for seed in [3,4]:
 for name,build,count in [('main',main.plan,8),('u1',u1.plan,5)]:
  jobs=build(seed);assert len(jobs)==count and len({j['id'] for j in jobs})==count
  for job in jobs:
   cmd=job['command'];assert cmd[cmd.index('--seed')+1]==str(seed)
   assert not any('seed2' in arg for arg in cmd)
   if '--resume-checkpoint' in cmd:
    prefix='u1-' if name=='u1' else ''
    assert cmd[cmd.index('--resume-checkpoint')+1]==f'checkpoints/{prefix}pick-seed{seed}-c-001/final.pt'
   if '--bc-checkpoint' in cmd:assert cmd[cmd.index('--bc-checkpoint')+1]==f'checkpoints/pick-seed{seed}-bc-001/final.pt'
  (out/f'{name}-seed{seed}.json').write_text(json.dumps(jobs,indent=2)+'\n')
plan=json.loads(Path('configs/final_evaluation_plan.json').read_text())
assert len(plan['entries'])==len({e['id'] for e in plan['entries']})==52
assert sum(len(e['methods']) for e in plan['entries'])==88
old_root=main.ROOT
with tempfile.TemporaryDirectory() as directory:
 main.ROOT=Path(directory)
 for hours,expected in [(20,7200),(22,0),(23,-3600)]:
  (main.ROOT/'research_state.json').write_text(json.dumps({'budget':{'gpu_hours_limit':24,'gpu_job_seconds_used':hours*3600,'final_evaluation_reserve_seconds':7200}}))
  assert main.training_seconds_left()==expected
 main.ROOT=old_root
(out/'result.json').write_text(json.dumps(dict(valid=True,old_plans_unchanged=True,new_seeds=[3,4],models=52,method_conditions=88,evaluation_reserve_seconds=7200,new_environment_interactions=0),indent=2)+'\n')
