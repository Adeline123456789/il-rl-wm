"""Portable frozen rfcl actor, without replay, critic or optimizer construction."""
import argparse
import hashlib
import json
import os
from pathlib import Path
os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE','false')
import flax.linen as nn
import flax.serialization
import jax
import jax.numpy as jnp
import numpy as np
from rfcl.agents.sac.networks import DiagGaussianActor
from rfcl.models.mlp import MLP
import torch


def export(checkpoint, output):
    state=json.loads((checkpoint/'state.json').read_text())
    blob=(checkpoint/'learner.msgpack').read_bytes()
    params=flax.serialization.msgpack_restore(blob)['actor']['params']
    weights=params['params']
    output.mkdir(parents=True,exist_ok=False)
    path=output/'actor.msgpack';path.write_bytes(flax.serialization.to_bytes(params))
    manifest=dict(config=state['config'],step=state['step'],
        obs_dim=int(weights['feature_extractor']['Dense_0']['kernel'].shape[0]),
        action_dim=int(weights['action_head']['bias'].shape[0]),
        source=str(checkpoint),source_sha256=hashlib.sha256(blob).hexdigest(),
        actor_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),bytes=path.stat().st_size,
        training_resume_supported=False,omitted=['critics','optimizers','replay','RNG'])
    (output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    return manifest


class FrozenPolicy:
    def __init__(self, directory):
        self.manifest=json.loads((directory/'manifest.json').read_text())
        blob=(directory/'actor.msgpack').read_bytes()
        assert hashlib.sha256(blob).hexdigest()==self.manifest['actor_sha256']
        params=jax.tree_util.tree_map(jnp.asarray,flax.serialization.msgpack_restore(blob))
        actor=DiagGaussianActor(MLP(features=self.manifest['config']['hidden_dims'],output_activation=nn.relu),
                                act_dims=self.manifest['action_dim'])
        self.act=jax.jit(lambda obs:actor.apply(params,obs,deterministic=True))

    def get_eval_action(self, obs):
        return torch.as_tensor(np.array(self.act(jnp.asarray(obs.detach().cpu().numpy()))),device=obs.device)

    def eval(self):
        return self

    def train(self):
        # Shared evaluation helper restores mode; this object has no learner.
        return self


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    print(json.dumps(export(args.checkpoint,args.output)))
