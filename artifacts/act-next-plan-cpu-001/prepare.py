from pathlib import Path
import json,hashlib,math
read=lambda p:json.loads(Path(p).read_text())
base=read('artifacts/act-t6-online-seed0-001/sac_config.json');r1=read('artifacts/act-t3-online-seed0-001/sac_config.json')
conditions=[]
for label,kind,init,bc in [('R0','mlp',None,None),('R1','mlp',None,r1['bc_checkpoint']),('R2','mlp',base['actor_init_checkpoint'],None),('R3','act','releases/act-c1-init-001/weights.pt',None)]:
 cfg=dict(base,actor_init_kind=kind,actor_init_checkpoint=init,bc_checkpoint=bc,output=f'artifacts/proposed-{label.lower()}-expert50-seed0',exp_name=f'proposed-{label.lower()}-expert50-seed0')
 conditions.append(dict(id=label,status='existing completed T6; reuse without rerun' if label=='R2' else 'not authorized; not queued; no results',config=cfg,existing_experiment='act-t6-online-seed0-001' if label=='R2' else None,
  pre_execution_gate='Current train_sac.py rejects expert replay with random initialization. Remove only that obsolete initialization requirement after CPU interface review; retain 50% batch split and explicit data audit.' if label=='R0' else None))
cost=read('artifacts/act-u2-final-review-cpu-001/analysis/result.json')['costs'];t4=cost['T4']['gpu_job_seconds'];t5=cost['T5']['gpu_job_seconds'];t6=cost['T6']['gpu_job_seconds']
plan=dict(status='proposal only; explicit new GPU authorization required',created='2026-10-19',source_task='ACT_NEXT_STAGE_TASK_2026-10-19.md',conditions=conditions,
 changed_factor='Actor initialization/architecture only within fixed50% expert replay; R3 includes architecture and cost change, not a pure weight-only intervention.',
 common=dict(task='PickCube-v1',reward='sparse',state_dim=42,action_dim=4,critic='fresh twin LayerNorm; same initial hashes',train_seed=0,train_interactions=262144,updates=64528,expert_samples=33038336,online_update_samples=33038336,validation_interactions=136000,validation_seed=10000,checkpoint_selection='fixed final',warmup='4096 uniform random actions; no U2 retention; no WM',teacher_deployment=False),
 budget=dict(authorized=False,mlp_measured_seconds=t6,transformer_estimate_seconds=t5*t6/t4,estimate_rule='MLP measured T6; R3 T5 times measured expert replay cost ratio T6/T4; then35% margin+60s, rounded upward. Extrapolation not measured R3 performance.',training_caps_seconds=dict(R0=1900,R1=1900,R3=6200),resource_smoke_cap_seconds=300,independent_evaluation_cap_seconds=1200,total_requested_seconds=11500,requested_gpu_hours=11500/3600,hardware='one RTX3090 24GB serial',includes_multiseed=False,includes_warmup_hypothesis=False),
 controls=['Before approval no GPU launch or automatic queue.','R0/C1 loading must not change fresh critic hashes; retain forked RNG.','Exact episode-reset matching; record all training and evaluation interactions.','Final seed20000 now used by old batch and cannot be used to select new methods. Preregister fresh held-out final seeds before new batch.','No same-seed threshold is statistical significance; fixed protocol then independent seed replication needs separate budget.'],
 stopping=['Stop on resource cap, nonfinite loss, invalid data provenance or initialization mismatch; preserve failure.','Run each registered condition once; no parameter search or best-checkpoint selection.','Compare fixed final and complete learning curves, early positive reward, teacher-free success and costs.','If R0 approximates R2, prefer explanation by expert data until replicated init benefit; do not add warmup or U2 to salvage scores.'])
Path('configs/act_next_initialization_controls.json').write_text(json.dumps(plan,indent=2)+'\n')
first=[json.loads(x) for x in Path('artifacts/act-matched-eval-001/act_k30_first/validation_episodes.jsonl').read_text().splitlines()]
ens=[json.loads(x) for x in Path('artifacts/act-matched-eval-001/act_k30_ensemble/validation_episodes.jsonl').read_text().splitlines()]
assert len(first)==len(ens)==40
out=dict(episodes=40,paired_episode_reset_seeds_equal=all((a['episode_index'],a['eval_seed'])==(b['episode_index'],b['eval_seed']) for a,b in zip(first,ens)),stages={})
for key in ['approached_once','is_grasped_once','lifted_once','is_obj_placed_once','success_once','success_at_end']:
 out['stages'][key]=dict(first=sum(bool(x[key]) for x in first),ensemble=sum(bool(x[key]) for x in ens),ensemble_only=sum(not bool(a[key]) and bool(b[key]) for a,b in zip(first,ens)),first_only=sum(bool(a[key]) and not bool(b[key]) for a,b in zip(first,ens)))
out['first_grasp_step_among_grasped']={label:dict(count=len(v),median=__import__('statistics').median(v)) for label,rows in [('first',first),('ensemble',ens)] for v in [[r['first_grasp_step'] for r in rows if r['first_grasp_step'] is not None]]}
out['limitation']='Matched initial states, divergent subsequent trajectories. No inference that a particular stage alone caused gain; no proof current-state student cannot approximate history behavior.'
p=Path('artifacts/act-temporal-stage-review-001');p.mkdir(exist_ok=False);(p/'result.json').write_text(json.dumps(out,indent=2)+'\n')
(p/'sources.json').write_text(json.dumps({str(x):hashlib.sha256(x.read_bytes()).hexdigest() for x in [Path('artifacts/act-matched-eval-001/act_k30_first/validation_episodes.jsonl'),Path('artifacts/act-matched-eval-001/act_k30_ensemble/validation_episodes.jsonl')]},indent=2)+'\n')
print(json.dumps(out,indent=2))
