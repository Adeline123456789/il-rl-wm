"""Real simulation smoke test; results are engineering evidence, not policy scores."""
import argparse
import inspect
import json
from pathlib import Path
import time

import gymnasium as gym
import mani_skill.envs
import numpy as np
import torch
from mani_skill.utils import gym_utils
from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env-id', default='PickCube-v1')
    parser.add_argument('--num-envs', type=int, default=4)
    parser.add_argument('--steps', type=int, default=100)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--control-mode', default='pd_ee_delta_pos')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    out = Path(args.output); out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    config = dict(env_id=args.env_id, num_envs=args.num_envs, obs_mode='state',
                  control_mode=args.control_mode, reward_mode='sparse', sim_backend='gpu',
                  reconfiguration_freq=1, render_mode=None)
    (out/'config.json').write_text(json.dumps(config, indent=2))
    raw = gym.make(args.env_id, **{k: v for k, v in config.items() if k != 'env_id'})
    env = ManiSkillVectorEnv(raw, ignore_terminations=True, record_metrics=True)
    obs, _ = env.reset(seed=args.seed)
    base = env.unwrapped
    source = {}
    for name in ['compute_sparse_reward', 'evaluate', 'get_reward']:
        method = getattr(base, name, None)
        if method:
            source[name] = {'path': inspect.getsourcefile(method), 'source': inspect.getsource(method)}
    (out/'environment-source.json').write_text(json.dumps(source, indent=2))
    started = time.perf_counter()
    rows, records = [], []
    lo = torch.as_tensor(env.single_action_space.low, device=obs.device)
    hi = torch.as_tensor(env.single_action_space.high, device=obs.device)
    for step in range(args.steps):
        action = lo + torch.rand(env.action_space.shape, device=obs.device) * (hi-lo)
        previous = obs.clone()
        obs, reward, terminated, truncated, info = env.step(action)
        true_next = obs.clone()
        mask = terminated | truncated
        if mask.any():
            assert 'final_observation' in info
            assert torch.equal(info['_final_observation'], mask)
            true_next[mask] = info['final_observation'][mask]
        assert torch.isfinite(obs).all() and torch.isfinite(reward).all()
        assert (action >= lo).all() and (action <= hi).all()
        assert not terminated.any(), 'ignore_terminations wrapper must clear returned terminations'
        transition_info = info.get('final_info', info)
        expected_reward = base.compute_sparse_reward(obs=previous, action=action, info=transition_info)
        assert torch.equal(reward, expected_reward), 'sparse reward disagrees with locked environment implementation'
        row = dict(step=step, reward=reward.cpu().tolist(), terminated=terminated.cpu().tolist(),
                   truncated=truncated.cpu().tolist(), success=transition_info['success'].cpu().tolist(),
                   next_observation_delta_l2=torch.linalg.vector_norm(true_next-previous, dim=-1).cpu().tolist())
        if mask.any():
            row['episode'] = {k: v.cpu().tolist() for k, v in info['final_info']['episode'].items()}
            row['reset_vs_final_delta_l2'] = torch.linalg.vector_norm(obs-true_next, dim=-1).cpu().tolist()
        rows.append(row)
        records.append(tuple(x.cpu().numpy() for x in [previous, action, reward, true_next, terminated, truncated]))
    elapsed = time.perf_counter()-started
    arrays = {name: np.stack([r[i] for r in records]) for i, name in enumerate(['observation','action','reward','next_observation','terminated','truncated'])}
    assert np.any(arrays['observation'] != arrays['next_observation']), 'No observed simulation state evolution'
    np.savez_compressed(out/'transitions.npz', **arrays)
    (out/'steps.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in rows))
    result = dict(purpose='smoke_only', env_id=args.env_id, seed=args.seed,
                  interactions=args.steps*args.num_envs, elapsed_seconds=elapsed,
                  steps_per_second=args.steps*args.num_envs/elapsed,
                  observation_shape=list(obs.shape), action_shape=list(action.shape),
                  action_low=lo.cpu().tolist(), action_high=hi.cpu().tolist(),
                  control_frequency=base.control_freq, simulation_frequency=base.sim_freq,
                  time_limit=gym_utils.find_max_episode_steps_value(raw),
                  completed_episodes=int(arrays['truncated'].sum()),
                  reward_dtype=str(arrays['reward'].dtype),
                  reward_values=np.unique(arrays['reward']).astype(float).tolist(),
                  torch_peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                  torch_peak_reserved_bytes=torch.cuda.max_memory_reserved())
    (out/'result.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    env.close()


if __name__ == '__main__':
    main()
