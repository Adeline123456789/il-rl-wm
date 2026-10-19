"""Explicit offline actor initialization without changing fresh SAC critic RNG."""
import hashlib
import math
from pathlib import Path
import torch


def initialize_actor(base_actor, checkpoint=None, kind='mlp', initial_log_std=None):
    device=next(base_actor.parameters()).device
    metadata=dict(kind=kind,checkpoint=checkpoint,initial_log_std=initial_log_std)
    actor=base_actor
    if checkpoint:
        path=Path(checkpoint);metadata['sha256']=hashlib.sha256(path.read_bytes()).hexdigest()
        # Both constructor randomness and loading stay outside the common critic RNG stream.
        with torch.random.fork_rng(devices=[device.index or 0] if device.type=='cuda' else []):
            state=torch.load(path,map_location='cpu',weights_only=False)
            if kind=='act':
                from scripts.act_sac_actor import ACTSACActor
                from scripts.act_teacher import ACTConfig
                assert state['state_dim']==base_actor.backbone[0].in_features
                assert state['action_dim']==base_actor.fc_mean.out_features
                actor=ACTSACActor(ACTConfig(**state['model_config']),state['state_dim'],state['action_dim'],
                    state['log_std_init'],state['action_epsilon']).to(device)
            else:
                assert kind=='mlp'
            actor.load_state_dict(state['actor'],strict=True)
        actor.requires_grad_(True)
    else:
        assert kind=='mlp', 'ACT transfer needs an explicit copied inference checkpoint'
    if initial_log_std is not None:
        assert -5<initial_log_std<2
        with torch.no_grad():
            actor.fc_logstd.weight.zero_()
            actor.fc_logstd.bias.fill_(math.atanh(2*(initial_log_std+5)/7-1))
    metadata['actor_parameters']=sum(p.numel() for p in actor.parameters())
    metadata['variance_trainable']=all(p.requires_grad for p in actor.fc_logstd.parameters())
    return actor,metadata
