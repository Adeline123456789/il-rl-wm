"""Recalculate U2/T5 review and T4/T6 protocol checks from saved evidence only."""
import argparse,csv,hashlib,json
from pathlib import Path
import numpy as np

def read(p):return json.loads(Path(p).read_text())
def lines(p):return [json.loads(s) for s in Path(p).read_text().splitlines()]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(exist_ok=False,parents=True)
 sources={};rows=[];behaviors={};costs={};metric_stats={}
 def tracked(p):sources[str(p)]=sha(p);return p
 for method,exp,drift,replay in [
 ('T5','act-t5-online-seed0-001','act-t5-online-drift-001','act-online-behavior-audit-002/act-t5-online-seed0-001'),
 ('U2','act-u2-retention-seed0-001','act-u2-online-drift-001','act-u2-online-behavior-001'),
 ('T6','act-t6-online-seed0-001','act-t6-online-drift-001','act-t6-evidence-audit-001'),
 ('T4','act-t4-online-seed0-001','act-t4-online-drift-001','act-online-behavior-audit-001/act-t4-online-seed0-001')]:
  root=Path('artifacts')/exp
  ev=lines(tracked(root/'validation_episodes.jsonl'));metrics=lines(tracked(root/'metrics.jsonl'))
  dr=read(tracked(Path('artifacts')/drift/'result.json'));manifest=read(tracked(root/'manifest.json'))
  result=read(tracked(root/'result.json'));resources=lines(tracked(root/'resources.jsonl'))
  assert result['training_interactions']==262144 and result['gradient_updates']==64528 and result['validation_interactions']==136000
  costs[method]=dict(training_interactions=262144,gradient_updates=64528,validation_interactions=136000,
    gpu_job_seconds=manifest['elapsed_seconds'],training_script_seconds=result['elapsed_seconds'],
    whole_gpu_peak_mib=max(r['memory_mib'] for r in resources),torch_peak_allocated_bytes=result['torch_peak_allocated_bytes'],
    evaluation_time_separation='Training-job total includes its 17 validation points; no separately logged GPU duration for each validation. Independent final evaluation reported separately.')
  data=np.load(tracked(Path('artifacts')/replay/'transitions.npz'))
  reward=data['reward'];assert len(reward)==262144
  pos=np.flatnonzero(reward>0)
  behavior=dict(positive_reward_transitions=len(pos),first_positive_transition_ordinal=int(pos[0]+1) if len(pos) else None,
    first_positive_vector_step_completed_interactions=int((pos[0]//16+1)*16) if len(pos) else None,
    positive_episode_ids=np.unique(data['episode_id'][pos]).tolist(),bins=[])
  for start,end in [(0,4096),(4096,16384),(16384,65536),(65536,131072),(131072,262144)]:
   idx=np.arange(start,end);positive=idx[reward[idx]>0]
   behavior['bins'].append(dict(start=start,end=end,positive_rewards=len(positive),grasped_observations=int(data['grasped'][idx].sum()),
    positive_episode_count=len(np.unique(data['episode_id'][positive])),action_std=data['action'][idx].std(0).tolist(),action_variance=data['action'][idx].var(0).tolist()))
  behaviors[method]=behavior
  metric_stats[method]={}
  for name in sorted(set(r['metric'] for r in metrics if r['metric'].startswith(('losses/','retention/')))):
   metric_stats[method][name]=[]
   for start,end in [(0,65536),(65536,262144)]:
    vals=[r for r in metrics if r['metric']==name and start<r['step']<=end]
    if vals:metric_stats[method][name].append(dict(start_exclusive=start,end_inclusive=end,records=len(vals),min=min(r['value'] for r in vals),max=max(r['value'] for r in vals),last=vals[-1]))
  for step in [0,16384,65536,131072,262144]:
   ep=[r for r in ev if r['step']==step];assert len(ep)==40
   d=next(r for r in dr['results'] if r['step']==step)
   met={r['metric']:r['value'] for r in metrics if r['step']==step and r['metric'].startswith(('losses/','retention/'))}
   rows.append(dict(method=method,step=step,episodes=40,success_once=sum(r['success_once'] for r in ep),
    success_at_end=sum(r['success_at_end'] for r in ep),grasped_once=sum(r['is_grasped_once'] for r in ep),
    placed_once=sum(r['is_obj_placed_once'] for r in ep),lifted_once=None,
    lift_missing_reason='Online validation logs do not record height/trajectory; cannot reconstruct lift counts from grasp/place booleans.',
    mean_return=float(np.mean([r['return'] for r in ep])),teacher_mse_per_dim=d['teacher_action_mse_per_dim'],
    mean_log_std=d['mean_log_std'],metrics=met))
 t4=read(tracked(Path('artifacts/act-t4-online-seed0-001/sac_config.json')));t6=read(tracked(Path('artifacts/act-t6-online-seed0-001/sac_config.json')))
 differences={k:[t4.get(k),t6.get(k)] for k in set(t4)|set(t6) if t4.get(k)!=t6.get(k)}
 assert set(differences)=={'output','exp_name','expert_dataset','expert_audit'},differences
 for method in ['T4','T5','T6','U2']:
  assert all(r['metrics'].get('losses/alpha',1)>0 for r in rows if r['method']==method)
 result=dict(rows=rows,real_replay=behaviors,costs=costs,logged_metric_ranges=metric_stats,T4_T6_config_differences=differences,
    classification='B: reduced expert-distribution mean drift, no sustained closed-loop success improvement; transient early success is retained. Not pure C: failure at49152 precedes65536 cutoff.',
    missing_measurements=['Online validation lift counts and first grasp time were not logged; do not impute.', 'No ground-truth Q calibration or causal identification of entropy/warmup/critic.', 'Soft Q includes entropy; its magnitude alone is not task-value overestimation.'],
    sources=sources,new_environment_interactions=0,new_training_updates=0,final_test_used_for_analysis=False)
 (a.output/'result.json').write_text(json.dumps(result,indent=2)+'\n')
 with (a.output/'stages.csv').open('w') as f:
  w=csv.DictWriter(f,fieldnames=['method','step','episodes','success_once','success_at_end','grasped_once','placed_once','lifted_once','mean_return']);w.writeheader();w.writerows({k:r[k] for k in w.fieldnames} for r in rows)
 print(json.dumps(dict(costs=costs,positive={k:{n:v[n] for n in ['positive_reward_transitions','first_positive_vector_step_completed_interactions']} for k,v in behaviors.items()},config_differences=differences),indent=2))
if __name__=='__main__':main()
