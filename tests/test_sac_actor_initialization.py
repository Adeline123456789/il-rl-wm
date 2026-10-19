from types import SimpleNamespace
import gymnasium as gym
import numpy as np
import torch
from scripts.train_sac import Actor,SoftQNetwork
from scripts.sac_actor_initialization import initialize_actor
from scripts.policy_bridge import fingerprint
from scripts.act_sac_actor import ACTSACActor
from scripts.act_teacher import ACTConfig


def env_spec():
    return SimpleNamespace(single_observation_space=gym.spaces.Box(-np.inf,np.inf,(42,),dtype=np.float32),single_action_space=gym.spaces.Box(-1,1,(4,),dtype=np.float32))


def test_initialization_preserves_critic_rng_and_enables_variance(tmp_path):
    env=env_spec();source=ACTSACActor(ACTConfig(num_queries=30)).eval()
    path=tmp_path/'act.pt';torch.save(dict(actor=source.state_dict(),model_config=vars(source.config),state_dim=42,action_dim=4,log_std_init=-2,action_epsilon=1e-6),path)
    hashes=[]
    for kind,checkpoint in [('mlp',None),('act',str(path))]:
        torch.manual_seed(19);base=Actor(env)
        actor,meta=initialize_actor(base,checkpoint,kind,-2)
        q=SoftQNetwork(env,layer_norm=True);hashes.append(fingerprint(q))
        assert meta['variance_trainable']
        assert torch.allclose(actor(torch.randn(8,42))[1],torch.full((8,4),-2.))
    assert hashes[0]==hashes[1]


def test_legacy_no_initialization_is_identity_and_rng_unchanged():
    base=Actor(env_spec());before=fingerprint(base);rng=torch.get_rng_state()
    result,_=initialize_actor(base)
    assert result is base and fingerprint(result)==before and torch.equal(rng,torch.get_rng_state())
