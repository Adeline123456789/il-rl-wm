import json,csv
from pathlib import Path
root=Path('artifacts');out=root/'f-u0-behavior-timeline-001'
behavior=json.loads((root/'f-u0-behavior-u0-fine-001/result.json').read_text())['bins']
control=json.loads((root/'f-u0-behavior-control-fine-001/result.json').read_text())['bins']
metrics={}
for r in map(json.loads,(root/'e6-u0-full-001/metrics.jsonl').read_text().splitlines()):
 metrics.setdefault(r['step'],{})[r['metric']]=r['value']
models={r['step']:r for r in map(json.loads,(root/'e6-u0-full-001/model_updates.jsonl').read_text().splitlines())}
rows=[]
for start in range(0,262144,8192):
 b=[r for r in behavior if r['start']==start];c=next(r for r in control if r['start']==start);end=start+8192
 assert sum(r['executions'] for r in b)==8192
 avg=lambda key:sum(r[key]*r['executions'] for r in b)/8192
 rows.append(dict(start=start,end=end,u0_positive_reward_fraction=avg('positive_reward_fraction'),u0_grasped_fraction=avg('grasped_fraction'),u0_next_goal_distance_m=avg('next_goal_distance_mean_m'),u0_il_fraction=next(r['execution_fraction'] for r in b if r['source']=='IL'),u0_random_sac_fraction=next(r['execution_fraction'] for r in b if r['source']=='random_SAC'),control_positive_reward_fraction=c['positive_reward_fraction'],control_next_goal_distance_m=c['next_goal_distance_mean_m'],q1_at_start=metrics.get(start,{}).get('losses/qf1_values'),q1_at_end=metrics[end]['losses/qf1_values'],task_value_loss_at_end=models[end]['value_loss'],wm_loss_at_end=models[end]['wm_loss']))
(out/'timeline.json').write_text(json.dumps(rows,indent=2)+'\n')
with (out/'timeline.csv').open('w',newline='') as f:
 writer=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');writer.writeheader();writer.writerows(rows)
print(json.dumps(rows[-4:],indent=2))
