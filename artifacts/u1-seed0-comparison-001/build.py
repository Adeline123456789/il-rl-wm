import csv,hashlib,json
from pathlib import Path
out=Path('artifacts/u1-seed0-comparison-001');source_paths=[];curves=[];comparison={}
for label,experiment,audit_path,drift_path,behavior_path in [
 ('U0','e6-u0-full-001','artifacts/e6-u0-full-001/checkpoint-audit.json','artifacts/f-u0-value-drift-portable-001/result.json','artifacts/f-u0-behavior-u0-fine-001/result.json'),
 ('U1','u1-pick-seed0-u0-001','artifacts/u1-pick-seed0-u0-audit-001/result.json','artifacts/u1-seed0-common-input-drift-001/result.json','artifacts/u1-seed0-behavior-001/result.json')]:
 folder=Path('artifacts')/experiment
 paths=[folder/name for name in ['validation_episodes.jsonl','metrics.jsonl','manifest.json','model.json','resources.jsonl']]+[Path(audit_path),Path(drift_path),Path(behavior_path)]
 source_paths.extend(paths)
 episodes=[json.loads(l) for l in paths[0].read_text().splitlines()]
 for step in sorted({r['step'] for r in episodes}):
  for method in ['il','sac','il_sac_wm','sac_wm']:
   selected=[r for r in episodes if r['step']==step and r['method']==method];assert len(selected)==16
   curves.append(dict(variant=label,training_seed=0,step=step,total_training_interactions=262144+step,method=method,episodes=16,success_once=sum(r['success_once'] for r in selected),mean_return=sum(r['return'] for r in selected)/16))
 manifest=json.loads(paths[2].read_text());model=json.loads(paths[3].read_text());audit=json.loads(Path(audit_path).read_text());drift=json.loads(Path(drift_path).read_text())
 behavior=json.loads(Path(behavior_path).read_text());last=[r for r in behavior['bins'] if r['end']==262144];assert sum(r['executions'] for r in last)==8192
 late={key:sum(r[key]*r['executions'] for r in last)/8192 for key in ['positive_reward_fraction','grasped_fraction','next_goal_distance_mean_m']}
 comparison[label]=dict(final_validation=[r for r in curves if r['variant']==label and r['step']==262144],job_seconds=manifest['elapsed_seconds'],trainable_parameters=model['trainable_parameters'],sampled_peak_gpu_mib=max(json.loads(l)['memory_mib'] for l in paths[4].read_text().splitlines()),execution_audit=audit,common_input_drift=drift['results'],last8192_behavior=late,first_success_action_source=behavior['first_success_action_source'])
with (out/'validation_curves.csv').open('w',newline='') as stream:
 writer=csv.DictWriter(stream,fieldnames=list(curves[0]),lineterminator='\n');writer.writeheader();writer.writerows(curves)
(out/'comparison.json').write_text(json.dumps(dict(training_seed=0,training_role='development',final_test_accessed=False,new_environment_interactions=0,results=comparison,limitations='One development seed, fixed validation episodes. Common input data are from U0; behavior data differ by actual trajectories. Association and Bellman consistency do not establish calibration, causal mediation, or cross-seed effects.'),indent=2)+'\n')
(out/'sources.json').write_text(json.dumps({str(path):hashlib.sha256(path.read_bytes()).hexdigest() for path in source_paths},indent=2)+'\n')
print(json.dumps({label:r['last8192_behavior'] for label,r in comparison.items()}))
