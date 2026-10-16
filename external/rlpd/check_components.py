"""CPU checks for the external adapter before any real GPU baseline run."""
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

# Keep this background compilation from competing for all training CPU cores.
os.sched_setaffinity(0, sorted(os.sched_getaffinity(0))[-4:])
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import flax.serialization
import gymnasium as gym
import jax
import numpy as np
from rfcl.agents.sac import SAC
from rfcl.agents.sac.config import TimeStep
from rfcl.utils.tools import combine
import torch

from external.rlpd.components import make_components, load_expert, physical_next_observation


def main():
    assert jax.devices()[0].platform == 'cpu'
    config = json.loads((ROOT/'external/rlpd/protocol.json').read_text())
    config.update(seed=0, num_envs=1, batch_size=4, buffer_size=64)
    env = SimpleNamespace(single_observation_space=gym.spaces.Box(-np.inf,np.inf,(42,),dtype=np.float32),
                          single_action_space=gym.spaces.Box(-1,1,(4,),dtype=np.float32))
    ac, cfg = make_components(config, np.zeros(42,np.float32), np.zeros(4,np.float32))
    learner = SAC('gym:cpu', ac, env, cfg=cfg)
    expert, manifest = load_expert(config)
    np.random.seed(123)
    half = cfg.batch_size*cfg.grad_updates_per_step//2
    one = expert.sample_random_batch(None, half)
    two = {k:v.copy() for k,v in one.items()}
    two['reward'] = np.zeros_like(two['reward'])
    batch = combine(one, two)
    for k in one:
        np.testing.assert_array_equal(batch[k][::2],one[k])
        np.testing.assert_array_equal(batch[k][1::2],two[k])
    new, metrics = learner.update_parameters(jax.random.PRNGKey(2), ac, TimeStep(**batch))
    assert int(new.critic.step)==20 and int(new.actor.step)==1 and int(new.temp.step)==1
    assert all(np.isfinite(np.array(x)).all() for x in jax.tree_util.tree_leaves(metrics))
    assert any(not np.array_equal(np.array(a),np.array(b)) for a,b in zip(jax.tree_util.tree_leaves(ac.critic.params),jax.tree_util.tree_leaves(new.critic.params)))
    assert new.critic(one['env_obs'][:2],one['action'][:2]).shape==(10,2)
    paths = [str(p) for p,_ in jax.tree_util.tree_flatten_with_path(new.critic.params)[0]]
    assert any('LayerNorm_2' in p for p in paths)
    restored = flax.serialization.from_bytes(ac, flax.serialization.to_bytes(new))
    for a,b in zip(jax.tree_util.tree_leaves(new),jax.tree_util.tree_leaves(restored)):
        np.testing.assert_array_equal(np.array(a),np.array(b))
    reset = torch.tensor([[1.,2.],[3.,4.]])
    final = torch.tensor([[5.,6.],[7.,8.]])
    result = physical_next_observation(reset, torch.tensor([False,False]), torch.tensor([True,False]), {'final_observation':final})
    torch.testing.assert_close(result,torch.tensor([[5.,6.],[3.,4.]]),rtol=0,atol=0)
    assert torch.equal(reset,torch.tensor([[1.,2.],[3.,4.]]))
    report = dict(valid=True, critic_updates=20, actor_updates=1, critics=10, layers_with_layernorm=3,
                  expert_transitions=expert.size, dataset_sha256=manifest['dataset_sha256'],
                  alternating_half_mixture=True, optimizer_roundtrip_exact=True, final_observation_boundary=True)
    Path(sys.argv[1]).write_text(json.dumps(report,indent=2))
    print(json.dumps(report))


if __name__=='__main__':main()
