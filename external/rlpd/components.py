"""Construction and audited data for the separate rfcl baseline, no main learner import."""
import hashlib
import json
from pathlib import Path

import flax.linen as nn
import h5py
import jax
import numpy as np
import optax
from rfcl.agents.sac import ActorCritic, SACConfig
from rfcl.agents.sac.networks import DiagGaussianActor
from rfcl.data.dataset import ReplayDataset
from rfcl.models.mlp import MLP


def make_components(config, obs, action):
    actor = DiagGaussianActor(
        MLP(features=config['hidden_dims'], output_activation=nn.relu),
        act_dims=action.shape[-1],
    )
    critic = MLP(features=config['hidden_dims'], output_activation=nn.relu, use_layer_norm=True)
    ac = ActorCritic.create(
        jax.random.PRNGKey(config['seed']), obs, action,
        actor=actor, critic_feature_extractor=critic,
        num_qs=config['num_qs'], num_min_qs=config['num_min_qs'],
        actor_optim=optax.adam(config['learning_rate']),
        critic_optim=optax.adam(config['learning_rate']),
        temperature_optim=optax.adam(config['learning_rate']),
        initial_temperature=config['initial_temperature'],
    )
    cfg = SACConfig(
        num_seed_steps=config['learning_starts'], seed_with_policy=False,
        replay_buffer_capacity=config['buffer_size'], batch_size=config['batch_size'],
        num_envs=config['num_envs'], steps_per_env=1,
        grad_updates_per_step=config['num_envs'] * config['critic_updates_per_transition'],
        actor_update_freq=config['critic_updates_per_transition'],
        num_qs=config['num_qs'], num_min_qs=config['num_min_qs'],
        discount=config['discount'], tau=config['tau'], backup_entropy=config['backup_entropy'],
        target_entropy=-action.shape[-1]/2,
    )
    return ac, cfg


def load_expert(config):
    audit_path = Path(config['dataset_audit'])
    for flag in ['INVALIDATED.json', 'INITIAL_OBSERVATION_CAVEAT.json']:
        assert not (audit_path.parent / flag).exists()
    audit = json.loads(audit_path.read_text())
    assert audit['alignment_validated']
    assert audit['env_info']['env_id'] == config['env_id']
    for key in ['reward_mode', 'control_mode']:
        assert audit['env_info']['env_kwargs'][key] == config[key]
    digest = hashlib.sha256(Path(config['dataset']).read_bytes()).hexdigest()
    assert digest == audit['dataset_sha256']
    ids = audit['train_episode_ids']
    assert not set(ids) & set(audit['validation_episode_ids'])
    # rfcl's sparse path reconstructs reward from success. Require exact equality
    # to the actual replay reward before using it, without changing either label.
    with h5py.File(config['dataset']) as source:
        for i in ids:
            trajectory = source[f'traj_{i}']
            np.testing.assert_array_equal(trajectory['rewards'][:], trajectory['success'][:])
    expert = ReplayDataset(config['dataset'], eps_ids=ids, reward_mode='sparse')
    assert expert.action_scale is None
    assert np.isfinite(expert.data['action']).all()
    assert abs(expert.data['action']).max() <= 1
    return expert, dict(dataset_sha256=digest, train_episode_ids=ids, transitions=expert.size,
                        raw_sparse_reward_exact=True, expert_fraction=0.5)


def physical_next_observation(next_obs, terminated, truncated, info):
    """The reset observation is for acting; the physical successor is for TD."""
    result = next_obs.clone()
    done = terminated | truncated
    if 'final_observation' in info:
        result[done] = info['final_observation'][done]
    else:
        assert not done.any()
    return result
