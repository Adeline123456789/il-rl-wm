import csv,hashlib,json
from pathlib import Path
import numpy as np
out=Path('artifacts/f1-all-probes-summary-001');probes=[];rows=[];sources={};prefixes={}
for name,variant,step in [('f1-u0-ranking-001','U0',262144),('f1-u0-precollapse-ranking-001','U0',196608),('f1-u0-step196608-prefix20-001','U0',196608),('f1-u1-step196608-prefix20-001','U1',196608)]:
 folder=Path('artifacts')/name
 m=json.loads((folder/'manifest.json').read_text());assert m['ended_at'] and m['exit_code']==0 and m['stop_reason'] is None
 r=json.loads((folder/'result.json').read_text());a=json.loads((folder/'analysis.json').read_text());counter=json.loads((folder/'interactions.json').read_text())
 assert r['new_training_interactions']==a['new_environment_interactions']==0
 with np.load(folder/'branches.npz') as raw:
  valid=raw['matching_error'].max(axis=(0,1,2))<=r['matching_tolerance'];prefix=raw['prefix_observations'].copy();assert prefix.shape[-1]==42
  np.testing.assert_allclose(prefix[...,39:42],prefix[...,26:29]-prefix[...,29:32],atol=1e-5,rtol=0)
  distances=np.linalg.norm(prefix[-1,:,39:42],axis=-1)
  prefixes[name]=prefix
  for mode,target in enumerate(['first_action_student','sequence_student']):
   actual=raw['returns'][mode].mean(0).T[valid];metrics=a['targets'][target]
   rows.append(dict(experiment=name,variant=variant,training_step=step,lineage_training_interactions=262144+step,prefix_steps=r['configuration']['prefix_steps'],target=target,valid_states=int(valid.sum()),rankable_pairs=metrics['rankable_pairs'],near_tie_pairs=metrics['near_tie_pairs'],pairwise_accuracy=metrics['weighted_pairwise_accuracy'],mean_regret=metrics['mean_regret'],tail_aware_tolerance=a['tail_aware_tolerance'],actual_return_min=float(actual.min()) if actual.size else None,actual_return_max=float(actual.max()) if actual.size else None,max_within_state_return_spread=float(np.ptp(actual,axis=1).max()) if actual.size else None))
 probes.append(dict(experiment=name,variant=variant,training_step=step,lineage_training_interactions=262144+step,prefix_steps=r['configuration']['prefix_steps'],job_seconds=m['elapsed_seconds'],diagnostic_interactions=counter['total_completed'],new_training_interactions=0,prefix_max_error=r['prefix_max_error'],prefix_end_goal_distances_m=distances.tolist(),prefix_end_grasped_count=int((prefix[-1,:,18]>.5).sum()),sequence_score_error_components=a['sequence_score_error_components']))
 for filename in ['manifest.json','result.json','analysis.json','interactions.json','branches.npz','score_decomposition.npz']:
  path=folder/filename;sources[str(path)]=hashlib.sha256(path.read_bytes()).hexdigest()
short=prefixes['f1-u0-step196608-prefix20-001'];long=prefixes['f1-u0-precollapse-ranking-001'][:len(short)]
summary=dict(probes=probes,targets=rows,u0_prefix20_vs_first20_of_prefix40_max_abs=float(np.abs(short-long).max()),new_environment_interactions=0,interpretation='Different training snapshots and actor-generated prefix states are separate probes. Zero rankable pairs means unidentifiable, not perfect ranking. Means and repeats are descriptive, not independent training seeds. This consolidation does not select new prefixes or change tolerance.')
(out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');(out/'sources.json').write_text(json.dumps(sources,indent=2)+'\n')
with (out/'ranks.csv').open('w',newline='') as stream:
 writer=csv.DictWriter(stream,fieldnames=list(rows[0]),lineterminator='\n');writer.writeheader();writer.writerows(rows)
