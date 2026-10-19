import hashlib,json,sys
from pathlib import Path
from types import SimpleNamespace
import gymnasium as gym
import numpy as np
import torch
sys.path.insert(0,str(Path.cwd()))
from scripts.train_sac import Actor,SoftQNetwork,Args,validate_args
from scripts.sac_actor_initialization import initialize_actor
from scripts.expert_replay import ExpertReplay
from scripts.policy_bridge import fingerprint
from dataclasses import fields
torch.set_num_threads(2);torch.manual_seed(0)
plan=json.load(open('configs/act_next_initialization_controls.json'));config=plan['conditions'][0]['config'];names={f.name for f in fields(Args)};args=Args(**{k:v for k,v in config.items() if k in names});validate_args(args)
assert args.actor_init_checkpoint is None and args.bc_checkpoint is None
spec=SimpleNamespace(single_observation_space=gym.spaces.Box(-np.inf,np.inf,(42,),dtype=np.float32),single_action_space=gym.spaces.Box(-1,1,(4,),dtype=np.float32))
actor=Actor(spec);mean_before={k:v.clone() for k,v in actor.state_dict().items() if not k.startswith('fc_logstd')}
actor,meta=initialize_actor(actor,None,'mlp',-2)
assert all(torch.equal(v,actor.state_dict()[k]) for k,v in mean_before.items())
before=fingerprint(actor);expert=ExpertReplay(args.expert_dataset,args.expert_audit,torch.device('cpu'));batch=expert.sample(args.batch_size//2)
audit=json.load(open(args.expert_audit));assert expert.manifest['train_episode_ids']==audit['train_episode_ids'];assert not set(audit['train_episode_ids'])&set(audit['validation_episode_ids'])
assert expert.size==55092 and batch['obs'].shape==(512,42)
with torch.no_grad():
 action,density,mean=actor.get_action(batch['obs']);_,logstd=actor(batch['obs'])
 assert torch.isfinite(action).all() and torch.isfinite(density).all() and torch.isfinite(mean).all()
 assert (action.abs()<=1).all();torch.testing.assert_close(logstd,torch.full_like(logstd,-2))
assert fingerprint(actor)==before
r=dict(scope='CPU interface and real expert-train data loading only; no simulator, optimizer update, transfer-success or RL-performance evidence',random_actor_mean_parameters_unchanged=True,actor_unchanged_after_sampling=True,actor_hash=before,expert_train_transitions=expert.size,sampled_expert_transitions=512,batch_size=1024,declared_expert_fraction=.5,training_episode_count=len(audit['train_episode_ids']),validation_episode_count=len(audit['validation_episode_ids']),sample_shape=list(batch['obs'].shape),finite_stochastic_density=True,log_std=-2,device='cpu',new_environment_interactions=0,new_training_updates=0,gpu_seconds=0,gpu_training_authorized=False,source_hashes={str(p):hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in ['scripts/train_sac.py','scripts/sac_actor_initialization.py','scripts/expert_replay.py','configs/act_next_initialization_controls.json',args.expert_audit]})
Path('artifacts/act-r0-interface-cpu-001/result.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r))
