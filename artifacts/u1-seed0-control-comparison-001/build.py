import csv,hashlib,json
from pathlib import Path
out=Path('artifacts/u1-seed0-control-comparison-001');rows=[];sources={}
for variant,experiment in [('C continuation','e3-sac-c-continue-full-001'),('C-LN continuation','u1-pick-seed0-c-continue-001'),('U0','e6-u0-full-001'),('U1','u1-pick-seed0-u0-001')]:
 source=Path('artifacts')/experiment/'validation_episodes.jsonl';data=[json.loads(l) for l in source.read_text().splitlines()];sources[str(source)]=hashlib.sha256(source.read_bytes()).hexdigest()
 for method in sorted({r.get('method','sac') for r in data}):
  selected=[r for r in data if r['step']==262144 and r.get('method','sac')==method];assert len(selected)==16
  rows.append(dict(variant=variant,training_seed=0,training_role='development',total_training_interactions=524288,method=method,episodes=16,success_once=sum(r['success_once'] for r in selected),mean_return=sum(r['return'] for r in selected)/16,source=str(source)))
(out/'result.json').write_text(json.dumps(dict(rows=rows,sources=sources,final_test_accessed=False,limitation='One development training seed; fixed validation16, not final100. C continuations share the respective C parent with U0/U1; normalization changes the C training trajectories.'),indent=2)+'\n')
with (out/'results.csv').open('w',newline='') as f:
 writer=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');writer.writeheader();writer.writerows(rows)
