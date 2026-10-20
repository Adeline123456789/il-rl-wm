"""Recompute existing final tables and control logs; no new env interaction."""
from pathlib import Path
import csv,json,statistics,hashlib
ROOT=Path(__file__).resolve().parents[2]
OUT=Path(__file__).resolve().parent
rows=list(csv.DictReader((ROOT/'artifacts/act-final-summary-004/conditions.csv').open()))
assert len(rows)==106 and all(r['status']=='completed' and int(r['episodes'])==100 for r in rows)
pick=[r for r in rows if r['variant']=='U1' and r['training_role']=='independent_repeat']
by={m:{int(r['training_seed']):r for r in pick if r['method']==m} for m in ['sac','sac_wm','il_sac_wm']}
paired=[]
for seed in sorted(by['sac']):
 a,b=by['sac'][seed],by['sac_wm'][seed]
 paired.append({'seed':seed,'success_pp_delta':float(b['success_once'])-float(a['success_once']),'mean_return_delta':float(b['mean_return'])-float(a['mean_return'])})
probes=json.loads((ROOT/'artifacts/f1-all-probes-summary-001/summary.json').read_text())
control=[]
for exp in ['act-r0-expert50-seed0-001','act-r1-expert50-seed0-001','act-t6-online-seed0-001']:
 records=[json.loads(s) for s in (ROOT/'artifacts'/exp/'validation_episodes.jsonl').read_text().splitlines()];step=max(r['step'] for r in records);r=[r for r in records if r['step']==step]
 control.append({'experiment':exp,'training_seed':0,'step':step,'episodes':len(r),'success_once':sum(x['success_once'] for x in r),'success_at_end':sum(x['success_at_end'] for x in r),'mean_return':statistics.mean(x['return'] for x in r)})
manifest=json.loads((ROOT/'artifacts/act-r3-expert50-seed0-001/manifest.json').read_text())
result={'completed_final_conditions':len(rows),'u1_paired_training_seeds':paired,'u1_mean_success_pp_delta':statistics.mean(r['success_pp_delta'] for r in paired),'u1_mean_return_delta':statistics.mean(r['mean_return_delta'] for r in paired),'diagnostic_interactions':sum(p['diagnostic_interactions'] for p in probes['probes']),'probe_target_summaries':probes['targets'],'controls':control,'r3_interrupted':{k:manifest[k] for k in ['ended_at','exit_code','stop_reason','elapsed_seconds','child_pid']},'new_environment_interactions':0,'new_gradient_updates':0,'gpu_jobs_started':0,'interpretation':'Descriptive only; no new causal evidence or independent training repeats.'}
(OUT/'existing_recomputed.json').write_text(json.dumps(result,indent=2))
print(json.dumps({k:v for k,v in result.items() if k!='probe_target_summaries'},indent=2))
