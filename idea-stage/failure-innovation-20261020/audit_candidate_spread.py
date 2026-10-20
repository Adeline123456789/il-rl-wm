"""Check a reviewer-suggested explanation on saved candidate arrays only."""
from pathlib import Path
import numpy as np,json
ROOT=Path(__file__).resolve().parents[2];OUT=Path(__file__).resolve().parent
ids=['f1-u0-ranking-001','f1-u0-precollapse-ranking-001','f1-u0-step196608-prefix20-001','f1-u1-step196608-prefix20-001'];rows=[]
for exp in ids:
 d=np.load(ROOT/'artifacts'/exp/'branches.npz');a=d['actions'][:,:,0,:] # state,candidate,action: normalized input actions
 pair=np.sqrt(np.mean((a[:,:,None,:]-a[:,None,:,:])**2,axis=-1))
 sac=a[:,1:,:] # registered candidates: index0 frozenIL, indices1..8 SAC
 sac_pair=np.sqrt(np.mean((sac[:,:,None,:]-sac[:,None,:,:])**2,axis=-1))
 sac_sv=[np.linalg.svd(x-x.mean(axis=0),compute_uv=False).tolist() for x in sac]
 sv=[np.linalg.svd(x-x.mean(axis=0),compute_uv=False).tolist() for x in a]
 real=d['real_states'][0,0,:,:,0,:].transpose(1,0,2) # first action target, repeat0, first resulting observation
 obs_pair=np.max(np.abs(real[:,:,None,:]-real[:,None,:,:]),axis=-1)
 rows.append({'experiment':exp,'first_action_max_pairwise_rms_per_state':pair.max(axis=(1,2)).tolist(),'centered_action_singular_values_per_state':sv,'sac_only_max_pairwise_rms_per_state':sac_pair.max(axis=(1,2)).tolist(),'sac_only_centered_singular_values_per_state':sac_sv,'first_resulting_obs_max_abs_candidate_difference_per_state':obs_pair.max(axis=(1,2)).tolist(),'note':'Normalized action-space dispersion and raw observation element differences only; no common physical units, no inferred long-horizon state contraction, no proof of useful outcome diversity.'})
(OUT/'candidate_spread_audit.json').write_text(json.dumps({'records':rows,'new_environment_interactions':0,'gpu_jobs_started':0,'new_gradient_updates':0},indent=2));print(json.dumps(rows,indent=2))
