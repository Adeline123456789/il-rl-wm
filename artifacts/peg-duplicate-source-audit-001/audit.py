import json,hashlib,collections,h5py,numpy as np
from pathlib import Path
out=Path('artifacts/peg-duplicate-source-audit-001')
p=Path('data/official/PegInsertionSide-v1/motionplanning/trajectory.h5')
m=json.loads(p.with_suffix('.json').read_text());groups=collections.defaultdict(list)
for e in m['episodes']:groups[e['episode_seed']].append(e)
rows=[]
with h5py.File(p) as f:
 for seed,es in groups.items():
  if len(es)<2:continue
  arrays=[]
  for e in es:
   g=f[f"traj_{e['episode_id']}"]
   h=hashlib.sha256()
   def add(name,item):
    if isinstance(item,h5py.Dataset):h.update(name.encode());h.update(item[0].tobytes())
   g['env_states'].visititems(add)
   arrays.append(dict(id=e['episode_id'],actions_sha256=hashlib.sha256(g['actions'][:].tobytes()).hexdigest(),first_state_sha256=h.hexdigest()))
  rows.append(dict(seed=seed,episodes=arrays))
result=dict(episodes=len(m['episodes']),unique_seeds=len(groups),nonempty_reset_options=sum(bool(e['reset_kwargs'].get('options')) for e in m['episodes']),duplicate_groups=rows,identical_actions_groups=sum(len({a['actions_sha256'] for a in r['episodes']})==1 for r in rows),identical_initial_state_groups=sum(len({a['first_state_sha256'] for a in r['episodes']})==1 for r in rows))
(out/'result.json').write_text(json.dumps(result,indent=2));print({k:v for k,v in result.items() if k!='duplicate_groups'})
