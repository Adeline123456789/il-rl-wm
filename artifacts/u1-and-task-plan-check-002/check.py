import json
from pathlib import Path
from scripts.run_pickcube_repeat import plan
from scripts.run_u1_repeat import plan as u1_plan
from scripts.audit_demos import split_episodes
out=Path('artifacts/u1-and-task-plan-check-002')
checks={}
for tag,data in [('pick','e0-pickcube-gpu-replay-full-003'),('stack','e0-stack-gpu-replay-full-001')]:
 audit=json.loads(Path('artifacts',data,'dataset-audit.json').read_text())
 eligible=[e for e in audit['episodes'] if e['episode_id'] not in audit['excluded_episode_ids']]
 train,val=split_episodes(eligible,42)
 assert sorted(train)==audit['train_episode_ids'] and sorted(val)==audit['validation_episode_ids']
 checks[tag+'_existing_split_unchanged']=True
for seed in [0,1,2]:
 jobs=u1_plan(seed)
 assert all('--critic-layer-norm' in job['command'] for job in jobs if job['command'][1]=='scripts/train_sac.py')
 (out/f'u1-seed{seed}.json').write_text(json.dumps(jobs,indent=2))
for tag,task in [('peg','PegInsertionSide-v1'),('stack','StackCube-v1')]:
 jobs=plan(0,task);(out/f'{tag}-seed0.json').write_text(json.dumps(jobs,indent=2))
 for job in jobs:
  command=job['command']
  if '--env-id' in command:assert command[command.index('--env-id')+1]==task
  for option in ['--demo-path','--expert-dataset']:
   if option in command:
    assert f'e0-{tag}-gpu-replay-full-'+('002' if tag=='peg' else '001') in command[command.index(option)+1]
checks['task_protocol_and_data_revision']=True
(out/'result.json').write_text(json.dumps(checks,indent=2))
