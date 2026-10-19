import os,time,json,csv,hashlib,collections,math
from pathlib import Path
p=Path('artifacts/batch-final-accounting-002');start=time.time()
s=dict(pid=os.getpid(),phase='running',exit_code=None);(p/'status.json').write_text(json.dumps(s))
try:
 rows=[];sources={};phase_counts=collections.defaultdict(lambda:collections.Counter());missing=[]
 for path in sorted(Path('artifacts').glob('*/manifest.json')):
  m=json.loads(path.read_text());assert m['ended_at'];d=path.parent;sources[str(path)]=hashlib.sha256(path.read_bytes()).hexdigest()
  resources=d/'resources.jsonl';samples=[json.loads(l) for l in resources.read_text().splitlines()] if resources.exists() else []
  if samples:assert all(x['index']==0 for x in samples)
  interactions=d/'interactions.json';counts=json.loads(interactions.read_text()) if interactions.exists() else None
  if counts is None:missing.append(m['experiment_id'])
  else:
   sources[str(interactions)]=hashlib.sha256(interactions.read_bytes()).hexdigest()
   if 'phases' in counts:
    for phase,values in counts['phases'].items():phase_counts[phase].update(values)
   else:
    assert 'replay_args' in counts and sum(e['interactions'] for e in counts['environments'])==counts['total_interactions']
    phase_counts['demo_replay_'+counts['replay_args']['sim_backend']]['reported_replay_interactions']+=counts['total_interactions']
  if resources.exists():sources[str(resources)]=hashlib.sha256(resources.read_bytes()).hexdigest()
  rows.append(dict(experiment_id=m['experiment_id'],exit_code=m['exit_code'],stop_reason=m['stop_reason'],gpu_job_seconds=m['elapsed_seconds'],whole_gpu_sampled_peak_mib=max((x['memory_mib'] for x in samples),default=None),reported_completed_transitions=counts.get('total_completed') if counts else None,reported_attempted_transitions=counts.get('total_attempted') if counts else None,reported_replay_interactions=counts.get('total_interactions') if counts else None,manifest=str(path)))
 audit=json.loads(Path('artifacts/act-final-delivery-audit-001/audit/result.json').read_text());total=sum(x['gpu_job_seconds'] for x in rows)
 assert len(rows)==audit['all_gpu_manifest_count'] and math.isclose(total,audit['all_gpu_job_seconds'],abs_tol=1e-6)
 result=dict(job_count=len(rows),gpu_job_seconds=total,gpu_job_hours=total/3600,remaining_under_24h_seconds=86400-total,whole_gpu_sampled_peak_mib=max(x['whole_gpu_sampled_peak_mib'] or 0 for x in rows),reported_interactions_by_phase=dict(phase_counts),missing_interaction_files=missing,final_test_completed=sum(x['reported_completed_transitions'] or 0 for x in rows if x['experiment_id'].startswith('e7-test-')),definition='GPU job elapsed seconds include failed attempts; interaction totals are reported counters only, with missing files explicit; legacy replay step counters stay separate from completed/attempted transition counters. Resource peak is sampled whole-card memory, not a continuous guarantee.',new_environment_interactions=0,new_training_updates=0)
 (p/'result.json').write_text(json.dumps(result,indent=2)+'\n');(p/'sources.json').write_text(json.dumps(sources,indent=2)+'\n')
 with (p/'jobs.csv').open('w') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 s.update(phase='completed',exit_code=0)
except BaseException as error:
 s.update(phase='failed',exit_code=1,error=repr(error));raise
finally:
 s['elapsed_seconds']=time.time()-start;(p/'status.json').write_text(json.dumps(s,indent=2));print(s)
