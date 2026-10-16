"""Isolated RLPD adaptation: locked rfcl JAX updates, ManiSkill real transitions.

No main learner import and no curriculum. See protocol.json for exact deviations.
Run only in .venv-rlpd through the shared experiment runner.
"""
import argparse
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import pickle
import random
import subprocess
import sys
import time

os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE', 'false')
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import flax.serialization
import gymnasium as gym
import jax
import jax.numpy as jnp
import mani_skill.envs
from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv
import numpy as np
from rfcl.agents.sac import SAC
from rfcl.agents.sac.config import TimeStep
from rfcl.utils.tools import combine, flatten_struct_to_dict
import torch

from external.rlpd.components import load_expert, make_components, physical_next_observation
from scripts.interaction_counter import CountedEnv, InteractionCounter
from scripts.sac_evidence import RawLabels, append_json


def to_numpy(value):
    return value.detach().cpu().numpy()


def save_snapshot(ac, learner, raw, rng, config, output, step, final=False):
    directory = ROOT / 'checkpoints' / config['exp_name'] / ('final' if final else f'step_{step}')
    directory.mkdir(parents=True, exist_ok=False)
    (directory/'learner.msgpack').write_bytes(flax.serialization.to_bytes(ac))
    count = learner.replay_buffer.size()
    np.savez(directory/'replay.npz', **{k:v[:count] for k,v in learner.replay_buffer.buffers.items()},
             **{k:v[:count] for k,v in raw.items()})
    with (directory/'rng.pkl').open('wb') as f:
        pickle.dump(dict(jax=np.array(rng), numpy=np.random.get_state(), python=random.getstate(),
                         torch=torch.get_rng_state(), cuda=torch.cuda.get_rng_state_all()), f)
    record = dict(step=step, critic_updates=int(ac.critic.step), actor_updates=int(ac.actor.step),
                  replay_pos=learner.replay_buffer.ptr, replay_full=learner.replay_buffer.full,
                  config=config, recovery='Learner/target/optimizers/RNG/replay saved; simulator state absent. Resume must be labeled a new segment.')
    (directory/'state.json').write_text(json.dumps(record, indent=2))
    files = {p.name:dict(bytes=p.stat().st_size, sha256=hashlib.sha256(p.read_bytes()).hexdigest())
             for p in directory.iterdir()}
    append_json(output/'checkpoints.jsonl', dict(step=step, final=final, directory=str(directory.relative_to(ROOT)), files=files))


def evaluate(ac, env, config, output, step):
    before = (random.getstate(), np.random.get_state(), torch.get_rng_state(), torch.cuda.get_rng_state_all())
    random.seed(config['validation_seed']);np.random.seed(config['validation_seed']);torch.manual_seed(config['validation_seed'])
    obs, _ = env.reset(seed=config['validation_seed'])
    start = time.perf_counter()
    for _ in range(config['max_episode_steps']):
        actions, _ = ac.act(jax.random.PRNGKey(0), ac.actor, jnp.asarray(to_numpy(obs)))
        obs, _, _, _, info = env.step(torch.as_tensor(np.array(actions), device=obs.device))
        if 'final_info' in info:
            for i in torch.where(info['_final_info'])[0].tolist():
                append_json(output/'validation_episodes.jsonl', dict(step=step, env=i,
                    **{k:float(v[i]) for k,v in info['final_info']['episode'].items()}))
    append_json(output/'evaluation_cost.jsonl', dict(step=step, seconds=time.perf_counter()-start, episodes=config['validation_episodes']))
    random.setstate(before[0]);np.random.set_state(before[1]);torch.set_rng_state(before[2]);torch.cuda.set_rng_state_all(before[3])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--exp-name', required=True)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    config = json.loads((ROOT/'external/rlpd/protocol.json').read_text())
    config.update(seed=args.seed, exp_name=args.exp_name, smoke=args.smoke)
    if args.smoke:
        config.update(total_timesteps=4096, learning_starts=1024, validation_episodes=4, validation_frequency=4096)
    assert sys.prefix == str(ROOT/'.venv-rlpd')
    assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT/'.cache/upstreams/rfcl',text=True).strip() == config['rfcl_commit']
    assert jax.devices()[0].platform == 'gpu'
    torch.set_num_threads(4)
    random.seed(args.seed);np.random.seed(args.seed);torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    output = args.output;output.mkdir(parents=True, exist_ok=True)
    assert not (output/'config.json').exists(), 'Use a new experiment ID'
    (output/'config.json').write_text(json.dumps(config, indent=2))
    counter = InteractionCounter(output/'interactions.json')
    env_kwargs = dict(obs_mode='state', sim_backend='gpu', render_mode='rgb_array',
        control_mode=config['control_mode'], reward_mode='sparse', max_episode_steps=config['max_episode_steps'], reconfiguration_freq=1)
    raw_env = RawLabels(CountedEnv(gym.make(config['env_id'], num_envs=config['num_envs'], **env_kwargs), counter, 'training'))
    env = ManiSkillVectorEnv(raw_env, config['num_envs'], ignore_terminations=True, record_metrics=True)
    eval_env = ManiSkillVectorEnv(CountedEnv(gym.make(config['env_id'], num_envs=config['validation_episodes'], **env_kwargs), counter, 'validation'), config['validation_episodes'], ignore_terminations=True, record_metrics=True)
    assert (env.single_action_space.low == -1).all() and (env.single_action_space.high == 1).all()
    ac, cfg = make_components(config, np.zeros(env.single_observation_space.shape, np.float32), np.zeros(env.single_action_space.shape, np.float32))
    expert, expert_manifest = load_expert(config)
    (output/'expert_manifest.json').write_text(json.dumps(expert_manifest, indent=2))
    (output/'learner_config.json').write_text(json.dumps(asdict(cfg), indent=2))
    # Only update_parameters and the buffer are used. rfcl's incomplete GPU
    # GymLoop is deliberately not called; no second set of learning equations.
    learner = SAC('gym:cpu', ac, env, cfg=cfg)
    parameters = sum(x.size for model in [ac.actor, ac.critic, ac.temp] for x in jax.tree_util.tree_leaves(model.params))
    assert parameters < 10_000_000
    (output/'model.json').write_text(json.dumps(dict(trainable_parameters=parameters, critic_ensemble=10, layer_norm=True), indent=2))
    shape = (learner.replay_buffer.buffer_size_per_env, config['num_envs'])
    raw = {k:np.zeros(shape, dtype=dtype) for k,dtype in [('terminated',bool),('truncated',bool),('success',bool),('episode_ids',np.int64),('timesteps',np.int32)]}
    episode_ids = np.arange(config['num_envs'], dtype=np.int64)
    timesteps = np.zeros(config['num_envs'], dtype=np.int32)
    rng = jax.random.PRNGKey(args.seed)
    obs, _ = env.reset(seed=args.seed)
    started = time.perf_counter();sampled = 0
    evaluate(ac, eval_env, config, output, 0)
    n = config['num_envs']
    for step in range(n, config['total_timesteps']+1, n):
        rng, action_key, sample_key, update_key = jax.random.split(rng, 4)
        if step <= config['learning_starts']:
            actions = 2*torch.rand(env.action_space.shape, device=obs.device)-1
        else:
            sampled_actions, _ = ac.sample(action_key, ac.actor, jnp.asarray(to_numpy(obs)))
            actions = torch.as_tensor(np.array(sampled_actions), device=obs.device)
        next_obs, reward, terminated, truncated, info = env.step(actions)
        physical_next = physical_next_observation(next_obs, terminated, truncated, info)
        assert torch.equal(reward.float(), raw_env.success.float())
        assert torch.isfinite(physical_next).all() and torch.isfinite(actions).all()
        pos = learner.replay_buffer.ptr
        learner.replay_buffer.store(env_obs=to_numpy(obs), next_env_obs=to_numpy(physical_next),
            action=to_numpy(actions), reward=to_numpy(reward), mask=np.ones(n, np.float32))
        for key in ['terminated','truncated','success']:
            raw[key][pos] = to_numpy(getattr(raw_env, key))
        raw['episode_ids'][pos] = episode_ids;raw['timesteps'][pos] = timesteps
        if 'final_info' in info:
            mask = info['_final_info']
            assert torch.equal(physical_next[mask], info['final_observation'][mask])
            append_json(output/'reset_checks.jsonl', dict(step=step, count=int(mask.sum()),
                final_vs_reset_l2=to_numpy((physical_next[mask]-next_obs[mask]).norm(dim=-1)).tolist(), bootstrap_stopped=0))
            if not (output/'first_boundary.npz').exists():
                np.savez(output/'first_boundary.npz', physical=to_numpy(info['final_observation'][mask]),
                         reset=to_numpy(next_obs[mask]), stored=learner.replay_buffer.buffers['next_env_obs'][pos,to_numpy(mask)])
            for i in torch.where(mask)[0].tolist():
                append_json(output/'train_episodes.jsonl', dict(step=step, env=i, episode_id=int(episode_ids[i]),
                    **{k:float(v[i]) for k,v in info['final_info']['episode'].items()}))
        timesteps += 1
        reset = to_numpy(terminated | truncated)
        episode_ids[reset] += n;timesteps[reset] = 0
        obs = next_obs
        if step > config['learning_starts']:
            half = cfg.batch_size*cfg.grad_updates_per_step//2
            batch = combine(learner.replay_buffer.sample_random_batch(sample_key, half), expert.sample_random_batch(sample_key, half))
            ac, metrics = learner.update_parameters(update_key, ac, TimeStep(**batch))
            sampled += half
            if step % 4096 == 0:
                values = {k:float(v) for k,v in flatten_struct_to_dict(metrics).items()}
                assert all(np.isfinite(v) for v in values.values()), values
                append_json(output/'metrics.jsonl', dict(step=step, critic_updates=int(ac.critic.step), actor_updates=int(ac.actor.step), elapsed_seconds=time.perf_counter()-started, **values))
        if step % config['validation_frequency'] == 0 or step == config['total_timesteps']:
            evaluate(ac, eval_env, config, output, step)
            save_snapshot(ac, learner, raw, rng, config, output, step, final=step==config['total_timesteps'])
    assert int(ac.critic.step) == (config['total_timesteps']-config['learning_starts'])*config['critic_updates_per_transition']
    assert int(ac.actor.step) == config['total_timesteps']-config['learning_starts']
    counter.flush()
    result = dict(training_interactions=config['total_timesteps'], critic_updates=int(ac.critic.step), actor_updates=int(ac.actor.step),
        expert_samples=sampled, online_samples=sampled, elapsed_seconds=time.perf_counter()-started,
        validation_interactions=counter.counts['validation']['completed_transitions'], final_test_evaluated=False)
    (output/'result.json').write_text(json.dumps(result, indent=2))
    env.close();eval_env.close()


if __name__ == '__main__':
    main()
