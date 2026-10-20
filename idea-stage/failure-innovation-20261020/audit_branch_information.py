"""Existing branches only. Retrospective descriptive targets, never regrade F1."""
from pathlib import Path
import numpy as np, json
ROOT=Path(__file__).resolve().parents[2]
OUT=Path(__file__).resolve().parent
experiments=['f1-u0-ranking-001','f1-u0-precollapse-ranking-001','f1-u0-step196608-prefix20-001','f1-u1-step196608-prefix20-001']
rows=[]
for exp in experiments:
 d=np.load(ROOT/'artifacts'/exp/'branches.npz')
 # [target, repeat, candidate, state, time]; rewards are observations, not generated.
 r=d['reward_traces']; g=d['returns'];success=d['success']
 recompute=np.sum(r*np.power(.99,np.arange(r.shape[-1])),axis=-1)
 assert np.max(np.abs(recompute-g))<1e-7
 targets=[]
 for m in range(2):
  avg=g[m].mean(axis=0); gaps=avg.max(axis=0)-avg.min(axis=0)
  mixed=np.any(success[m],axis=1)&~np.all(success[m],axis=1)
  targets.append({'mode':m,'state_count':avg.shape[-1],'repeat_count':r.shape[1],'observed_max_candidate_return_gap':float(gaps.max()),'states_with_any_observed_return_gap_gt_1e-9':int(np.sum(gaps>1e-9)),'states_with_any_observed_return_gap_gt_1':int(np.sum(gaps>1+1e-9)),'mixed_observed_success_states_each_repeat':mixed.sum(axis=1).tolist(),'all_candidates_reward_trace_equal_states_each_repeat':np.all(r[m]==r[m,:,0:1,:,:],axis=(1,3)).sum(axis=1).tolist(),'repeat_reward_trace_max_abs_difference':float(np.max(np.abs(r[m,0]-r[m,1])))})
 rows.append({'experiment':exp,'targets':targets,'full_continuation_state_recorded':False,'coalescence_provable_from_saved_arrays':False,'available_real_states_shape':list(d['real_states'].shape),'note':'Only first2 resulting42-dim observations saved; neither full simulator/controller/RNG state nor full continuation states. Equal reward traces do not imply exact state coupling.'})
result={'experiments':rows,'new_environment_interactions':0,'new_gradient_updates':0,'gpu_jobs_started':0,'interpretation':'Post-hoc descriptive information audit only. Does not revise preregistered tolerance, score, outcome definition or declare precise causal diagnosis.'}
(OUT/'branch_information_audit.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
