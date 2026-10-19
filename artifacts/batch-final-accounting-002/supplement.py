import json,hashlib,os,time
from pathlib import Path
p=Path('artifacts/batch-final-accounting-002');start=time.time();rows=[];sources={}
missing=json.loads((p/'result.json').read_text())['missing_interaction_files']
for name in missing:
 d=Path('artifacts')/name;q=d/'result.json';row=dict(experiment_id=name)
 if q.exists():
  r=json.loads(q.read_text());sources[str(q)]=hashlib.sha256(q.read_bytes()).hexdigest()
  if 'new_environment_interactions' in r:
   assert r['new_environment_interactions']==0;row.update(classification='explicit_offline_zero',count=0)
  elif 'validation_interactions' in r:
   row.update(classification='reported_validation',count=r['validation_interactions'])
  else:
   assert r['purpose']=='smoke_only';row.update(classification='reported_environment_smoke',count=r['interactions'])
 elif name=='act-independent-eval-smoke-001':
  children=sorted(d.glob('*/interactions.json'));assert len(children)==6
  row.update(classification='nested_validation_counters',count=sum(json.loads(c.read_text())['total_completed'] for c in children))
  for c in children:sources[str(c)]=hashlib.sha256(c.read_bytes()).hexdigest()
 elif name=='act-t2-online-seed0-001':
  q=d/'stderr.log';assert "No module named 'gymnasium'" in q.read_text();sources[str(q)]=hashlib.sha256(q.read_bytes()).hexdigest();row.update(classification='failed_before_environment_import',count=0)
 else:
  assert name in ['e0-peg-replay-001','e0-peg-replay-fixed-001','e1-bc-full-001'];row.update(classification='uninstrumented_legacy_attempt',count=None)
  q=d/'validation_episodes.jsonl'
  if q.exists():
   records=[json.loads(l) for l in q.read_text().splitlines()];row['observed_complete_episode_steps_lower_bound']=sum(r['episode_len'] for r in records);row['unobserved_tail_unknown']=True;sources[str(q)]=hashlib.sha256(q.read_bytes()).hexdigest()
 rows.append(row)
result=dict(rows=rows,known_additional_reported_interactions=sum(r['count'] or 0 for r in rows),unresolved=[r['experiment_id'] for r in rows if r['count'] is None],definition='Only jobs without a top-level counter enter this supplement. Explicit offline zeros, nested counters and result counters are distinguished. Three legacy attempts are not assigned invented exact counts.',new_environment_interactions=0,new_training_updates=0)
(p/'supplement.json').write_text(json.dumps(result,indent=2)+'\n');(p/'supplement_sources.json').write_text(json.dumps(sources,indent=2)+'\n');(p/'supplement_status.json').write_text(json.dumps(dict(pid=os.getpid(),phase='completed',exit_code=0,elapsed_seconds=time.time()-start)));print(result)
