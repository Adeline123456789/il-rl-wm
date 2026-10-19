"""Evidence helpers for the official ManiSkill SAC adaptation, without a second learner."""
import hashlib
import json
from pathlib import Path
import random

import gymnasium as gym
import numpy as np
import torch


class RawLabels(gym.Wrapper):
    """Preserve task labels before the vector wrapper ignores success termination."""
    def step(self, action):
        result = self.env.step(action)
        self.terminated = result[2].clone()
        self.truncated = result[3].clone()
        self.success = result[4]['success'].clone()
        return result


def append_json(path, record):
    with Path(path).open('a') as stream:
        stream.write(json.dumps(record, allow_nan=False) + '\n')


def evaluate_policy(actor, env, args, output, step, method=None, episode_filename='validation_episodes.jsonl'):
    # Evaluation must not consume training RNG streams.
    states = (random.getstate(), np.random.get_state(), torch.get_rng_state(), torch.cuda.get_rng_state_all())
    random.seed(args.eval_seed);np.random.seed(args.eval_seed);torch.manual_seed(args.eval_seed)
    actor.eval()
    obs, _ = env.reset(seed=args.eval_seed)
    episodes = 0
    event_seen = {}
    matched_resets = getattr(args, 'matched_eval_resets', False)
    reset_seed = args.eval_seed
    for eval_step in range(args.num_eval_steps):
        with torch.no_grad():
            obs, _, _, _, info = env.step(actor.get_eval_action(obs))
        actual_info = info.get('final_info', info)
        for key in ['is_grasped', 'is_obj_placed', 'is_robot_static']:
            if key in actual_info:
                if key not in event_seen:
                    event_seen[key] = torch.zeros(env.num_envs, dtype=torch.bool, device=obs.device)
                event_seen[key] |= actual_info[key].bool()
        if 'final_info' in info:
            mask = info['_final_info']
            for i in torch.where(mask)[0].tolist():
                append_json(output / episode_filename, dict(step=step, episode=episodes, env=i, **({} if method is None else dict(method=method)), **(dict(eval_seed=reset_seed) if matched_resets else {}), **{k:float(v[i]) for k,v in info['final_info']['episode'].items()}, **{k+'_once':bool(v[i]) for k,v in event_seen.items()}))
                episodes += 1
            for value in event_seen.values():
                value[mask] = False
            if matched_resets and eval_step+1 < args.num_eval_steps:
                assert mask.all(), 'This matched evaluation requires synchronized full episodes'
                reset_seed = args.eval_seed + episodes
                obs, _ = env.reset(seed=reset_seed)
    assert episodes == args.num_eval_envs * args.num_eval_steps // args.max_episode_steps
    actor.train()
    random.setstate(states[0]); np.random.set_state(states[1]); torch.set_rng_state(states[2]); torch.cuda.set_rng_state_all(states[3])
    return args.num_eval_envs * args.num_eval_steps


def save_snapshot(output, run_name, step, updates, actor, q1, q2, qt1, qt2, actor_opt, q_opt, log_alpha, alpha_opt, replay, args, il, final=False, u0=None):
    directory = Path('checkpoints') / run_name
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / ('final.pt' if final else f'step_{step}.pt')
    # Save populated replay prefix including labels and chronology for WM sequences.
    count = replay.per_env_buffer_size if replay.full else replay.pos
    replay_state = {k:v[:count].cpu() for k,v in vars(replay).items() if isinstance(v, torch.Tensor)}
    replay_state.update(pos=replay.pos, full=replay.full, num_envs=replay.num_envs, capacity=replay.per_env_buffer_size)
    state = dict(actor=actor.state_dict(), qf1=q1.state_dict(), qf2=q2.state_dict(), qf1_target=qt1.state_dict(), qf2_target=qt2.state_dict(), actor_optimizer=actor_opt.state_dict(), q_optimizer=q_opt.state_dict(), log_alpha=None if log_alpha is None else log_alpha.detach(), alpha_optimizer=None if alpha_opt is None else alpha_opt.state_dict(), replay=replay_state, global_step=step, global_update=updates, config=vars(args), frozen_il=None if il is None else il.state_dict(), rng=dict(python=random.getstate(), numpy=np.random.get_state(), torch=torch.get_rng_state(), cuda=torch.cuda.get_rng_state_all()), recovery='Learner/RNG/replay saved. Simulator/controller/partial episode state is not serialized; resumption must be a labeled new segment, not claimed bit-exact.')
    state['lineage_training_interactions']=getattr(args,'inherited_training_interactions',0)+step
    if u0 is not None:state['u0']=u0
    temp = path.with_suffix('.tmp')
    torch.save(state, temp); temp.replace(path)
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(2**20), b''): h.update(block)
    append_json(output / 'checkpoints.jsonl', dict(step=step, final=final, path=str(path), bytes=path.stat().st_size, sha256=h.hexdigest()))


def restore_new_segment(path, actor, q1, q2, qt1, qt2, actor_opt, q_opt, log_alpha, alpha_opt, replay, il, output, args):
    state=torch.load(path,map_location='cpu')
    assert state['config'].get('critic_layer_norm',False)==args.critic_layer_norm, 'Resume must preserve critic parameterization'
    for key in ['env_id','control_mode','max_episode_steps','gamma','reconfiguration_freq','bootstrap_at_done']:
        assert state['config'][key]==getattr(args,key), f'Resume protocol mismatch: {key}'
    assert state['replay']['num_envs']==replay.num_envs and state['replay']['capacity']==replay.per_env_buffer_size
    for model,key in [(actor,'actor'),(q1,'qf1'),(q2,'qf2'),(qt1,'qf1_target'),(qt2,'qf2_target')]:model.load_state_dict(state[key])
    actor_opt.load_state_dict(state['actor_optimizer']);q_opt.load_state_dict(state['q_optimizer'])
    with torch.no_grad():log_alpha.copy_(state['log_alpha'])
    alpha_opt.load_state_dict(state['alpha_optimizer'])
    for key,value in state['replay'].items():
        if isinstance(value,torch.Tensor):getattr(replay,key)[:len(value)].copy_(value)
    replay.pos=state['replay']['pos'];replay.full=state['replay']['full']
    if il is not None:
        assert all(torch.equal(v.cpu(),state['frozen_il'][k]) for k,v in il.state_dict().items())
    parent_steps=state.get('lineage_training_interactions',state['global_step'])
    (output/'resume.json').write_text(json.dumps(dict(parent=str(path),sha256=hashlib.sha256(Path(path).read_bytes()).hexdigest(),
        inherited_training_interactions=parent_steps,parent_gradient_updates=state['global_update'],
        recovery='New simulator segment; learner/optimizers/temperature/replay retained. Fresh configured seed, not an exact interrupted trajectory/RNG continuation.'),indent=2))
    return parent_steps,int(state['replay']['episode_ids'].max())+1,state.get('u0')
