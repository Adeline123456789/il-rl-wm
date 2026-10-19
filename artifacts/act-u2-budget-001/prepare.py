import json,math,os,time
from pathlib import Path
out=Path('artifacts/act-u2-budget-001')
def read(p): return json.loads(Path(p).read_text())
def metric(folder,key):
 rows=[json.loads(x) for x in (folder/'metrics.jsonl').read_text().splitlines()]
 assert all(math.isfinite(x['value']) for x in rows)
 return next(x['value'] for x in reversed(rows) if x['metric']==key)
try:
 root=Path('artifacts/act-u2-retention-smoke-001')
 deadline=time.monotonic()+300
 while not read(root/'manifest.json')['ended_at']:
  assert time.monotonic()<deadline
  time.sleep(2)
 m=read(root/'manifest.json');assert m['exit_code']==0 and m['stop_reason'] is None
 r=read(root/'result.json');assert r['training_interactions']==8192 and r['gradient_updates']==1040 and r['validation_interactions']==16000
 retention=read(root/'retention_result.json');assert retention['updates']==1040 and retention['samples']==66560 and retention['final_coefficient']==8.75
 labels=read(root/'retention_manifest.json');assert labels['states']==55092 and labels['teacher_frozen_verified'] and not labels['expert_transitions_in_critic']
 assert read(root/'sampling.json')['expert_samples']==0
 initial=read(root/'actor_initialization.json');baseline=Path('artifacts/act-t5-online-seed0-001')
 assert initial==read(baseline/'actor_initialization.json')
 a=read(root/'sac_config.json');b=read(baseline/'sac_config.json');assert all(a[k]==v for k,v in b.items() if k not in ['output','exp_name','total_timesteps'])
 old=read(baseline/'manifest.json')['elapsed_seconds']
 old_update=metric(baseline,'time/total_update_time');old_rollout=metric(baseline,'time/total_rollout_time')
 new_update=metric(root,'time/total_update_time')
 projected=new_update/1040*15360+old_update/64528*(64528-15360)+old_rollout+(old-old_update-old_rollout)+labels['initialization_wall_seconds']
 cap_estimate=projected*1.1+60
 used=sum(read(f)['elapsed_seconds'] for f in Path('artifacts').glob('*/manifest.json'))
 assert cap_estimate<=4500 and 86400-used-7200>=4500
 original=read('artifacts/act-online-budget-002/plan.json')
 job=original['jobs'][3].copy();job['id']='act-u2-retention-seed0-001';job['seconds']=4500
 job['command']=[x.replace('act-t5-online-seed0-001',job['id']) for x in job['command']]+['--retention-config','configs/act_u2_retention.json']
 plan=dict(fits=True,jobs=[job],fixed_critic_hashes=original['fixed_critic_hashes'],reserve_seconds=7200,used_seconds=used,research_remaining_seconds=86400-used-7200,estimate_seconds=cap_estimate,projection_seconds=projected,smoke_update_seconds=new_update,baseline_update_seconds=old_update,baseline_rollout_seconds=old_rollout,active_retention_updates=15360,baseline_updates=64528-15360,selection='fixed_final; resource/interface gate only; no score-based decision',hypothesis='U2_CHANGE.md')
 (out/'plan.json').write_text(json.dumps(plan,indent=2)+'\n')
 (out/'exit.json').write_text(json.dumps(dict(pid=os.getpid(),exit_code=0)))
except BaseException:
 (out/'exit.json').write_text(json.dumps(dict(pid=os.getpid(),exit_code=1)))
 raise
