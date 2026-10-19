from pathlib import Path
import json,time,statistics,hashlib,os
p=Path('artifacts/act-final-throughput-review-001');p.mkdir(exist_ok=False)
read=lambda f:json.loads(Path(f).read_text())
plan=read('configs/act_final_evaluation_plan.json');s=read('artifacts/act-final-evaluation-batch-003/status.json');launch=read('artifacts/act-final-evaluation-launch-003/status.json')
finished=[];groups={}
for e in plan['entries']:
 f=Path('artifacts')/('e7-test-'+e['id']+'-001')/'manifest.json'
 if not f.exists():continue
 m=read(f)
 if not m.get('ended_at') or m.get('exit_code')!=0:continue
 key=e['framework']+'_'+str(len(e['methods']))+'_'+e['task'];groups.setdefault(key,[]).append(m['elapsed_seconds']);finished.append(e['id'])
plain=statistics.mean(groups['main_1_PickCube-v1']);multi=statistics.mean(groups['main_4_PickCube-v1'])
rows=[]
for e in plan['entries']:
 if e['id'] in finished:continue
 horizon={'PickCube-v1':200,'StackCube-v1':300,'PegInsertionSide-v1':400}[e['task']]
 key=e['framework']+'_'+str(len(e['methods']))+'_'+e['task']
 if key in groups:low=high=statistics.mean(groups[key]);basis='same framework/method-count/task completed jobs'
 elif e['framework']=='main':low=high=(multi if len(e['methods'])==4 else plain)*horizon/200;basis='Pick measured job time scaled by horizon; startup overestimate'
 elif e['framework']=='act':low=plain;high=plain*1.5;basis='same simulator/horizon; model-specific synchronized latency recorded below; wall range unvalidated'
 else:low=plain;high=600;basis='RLPD compilation overhead unmeasured in final entry;600s existing per-job cap upper bound'
 rows.append(dict(id=e['id'],seconds_low=low,seconds_high=high,basis=basis))
smoke={}
for d in Path('artifacts/act-independent-eval-smoke-001').iterdir():
 f=d/'result.json'
 if f.exists():
  r=read(f);smoke[d.name]={k:v for k,v in r.items() if k.startswith('latency_')}
ledger=read('research_state.json')['budget'];elapsed=time.time()-launch['started_at'];remaining=max(0,s['effective_limit_seconds']-elapsed)
out=dict(captured_at=time.time(),pid=os.getpid(),completed_entries=len(finished),registered_entries=len(plan['entries']),registered_conditions=sum(len(e['methods']) for e in plan['entries']),
 plan_sha256=hashlib.sha256(Path('configs/act_final_evaluation_plan.json').read_bytes()).hexdigest(),ledger=ledger,current=s['current'],groups_seconds=groups,
 resumed_batch_elapsed_seconds=elapsed,remaining_original_batch_seconds=remaining,remaining_projection_low=sum(x['seconds_low'] for x in rows),remaining_projection_high=sum(x['seconds_high'] for x in rows),
 pending=rows,validation_smoke_model_latency=smoke,selection_changed=False,
 limitation='Forecast uses timings only, no test scores. Current in-flight job is charged at full predicted duration, conservative. Final model variants vary; range is not guarantee. Fixed registered order retained; budget-stopped/unrun scores remain null. No balanced subset selected after seeing final results.')
(p/'result.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps({k:out[k] for k in ['completed_entries','remaining_original_batch_seconds','remaining_projection_low','remaining_projection_high']},indent=2))
