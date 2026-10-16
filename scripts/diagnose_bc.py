"""Frozen BC diagnostics: expert replay vs BC from source states and independent resets.

Diagnostic-only interactions never enter training replay. This is not exact
arbitrary-state branch restore; it recreates initial states as the official demo
replayer does and verifies initial observation alignment before interpreting it.
"""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import gymnasium as gym
import h5py
import numpy as np
import torch
import mani_skill.envs
from scripts.train_bc import Actor
from scripts.interaction_counter import InteractionCounter, CountedEnv


def initial_state(group):
    return {k:initial_state(v) if isinstance(v,h5py.Group) else v[0] for k,v in group.items()}


def stack_states(states):
    return {k:stack_states([s[k] for s in states]) if isinstance(states[0][k],dict) else np.stack([s[k] for s in states]) for k in states[0]}


def main():
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--audit',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--episodes',type=int,default=16);args=p.parse_args()
    torch.set_num_threads(4);torch.manual_seed(2026);np.random.seed(2026)
    args.output.mkdir(exist_ok=True,parents=True)
    assert not (args.output/'diagnostic.json').exists()
    split=json.loads(args.audit.read_text());assert split['alignment_validated']
    ids=np.random.default_rng(2026).choice(split['validation_episode_ids'],size=args.episodes,replace=False).tolist()
    metadata=json.loads(Path(split['dataset']).with_suffix('.json').read_text());by_id={e['episode_id']:e for e in metadata['episodes']}
    with h5py.File(split['dataset']) as f:
        states=[initial_state(f[f'traj_{i}']['env_states']) for i in ids]
        first=np.stack([f[f'traj_{i}']['obs'][0] for i in ids])
        actions=[f[f'traj_{i}']['actions'][:] for i in ids]
    n=args.episodes;lengths=np.array([len(a) for a in actions]);seeds=[int(by_id[i]['episode_seed']) for i in ids]
    policy=Actor(first.shape[1],actions[0].shape[1]).cuda();policy.load_state_dict(torch.load(args.checkpoint,map_location='cuda')['actor']);policy.eval();policy.requires_grad_(False)
    counter=InteractionCounter(args.output/'interactions.json');results={}
    for mode in ['expert_source_initial','bc_source_initial','bc_independent_reset']:
        env=CountedEnv(gym.make('PickCube-v1',num_envs=n,obs_mode='state',control_mode='pd_ee_delta_pos',sim_backend='physx_cuda',reward_mode='sparse',max_episode_steps=200,reconfiguration_freq=1),counter,mode)
        obs,_=env.reset(seed=seeds if mode!='bc_independent_reset' else 10000)
        alignment=None
        if mode!='bc_independent_reset':
            env.unwrapped.set_state_dict(stack_states(states));obs=env.unwrapped.get_obs()
            alignment=float(np.abs(obs.cpu().numpy()-first).max())
        seen={k:np.zeros(n,dtype=bool) for k in ['success','is_grasped','is_obj_placed','is_robot_static']}
        returns=np.zeros(n);last_success=np.zeros(n,dtype=bool);last_distance=np.zeros(n);trace={k:[] for k in ['obs','action','reward','success','is_grasped','is_obj_placed','is_robot_static','goal_distance']}
        steps=int(lengths.max()) if mode=='expert_source_initial' else 200
        with torch.no_grad():
            for t in range(steps):
                if mode=='expert_source_initial':
                    action=torch.as_tensor(np.stack([a[t] if t<len(a) else np.zeros(a.shape[1]) for a in actions]),dtype=torch.float32,device='cuda')
                    valid=t<lengths
                else:
                    action=policy(obs);valid=np.ones(n,dtype=bool)
                trace['obs'].append(obs.cpu().numpy());trace['action'].append(action.cpu().numpy())
                obs,reward,_,_,info=env.step(action)
                reward=reward.float().cpu().numpy();trace['reward'].append(reward);returns[valid]+=reward[valid]
                for key in seen:
                    values=info[key].cpu().numpy().astype(bool);seen[key][valid]|=values[valid];trace[key].append(values)
                last_success[valid]=info['success'].cpu().numpy()[valid]
                distance=(env.unwrapped.goal_site.pose.p-env.unwrapped.cube.pose.p).norm(dim=-1).cpu().numpy()
                last_distance[valid]=distance[valid];trace['goal_distance'].append(distance)
        trace['obs'].append(obs.cpu().numpy())
        np.savez_compressed(args.output/f'{mode}.npz',**{k:np.stack(v) for k,v in trace.items()})
        results[mode]=dict(initial_observation_max_abs_error=alignment,source_episode_ids=ids if mode!='bc_independent_reset' else None,episodes=[dict(index=i,source_seed=seeds[i] if mode!='bc_independent_reset' else None,success_once=bool(seen['success'][i]),success_at_end=bool(last_success[i]),return_=float(returns[i]),goal_distance_at_end=float(last_distance[i]),**{k+'_once':bool(v[i]) for k,v in seen.items() if k!='success'}) for i in range(n)])
        env.close();counter.flush()
        (args.output/'diagnostic.json').write_text(json.dumps(dict(checkpoint=str(args.checkpoint),audit=str(args.audit),selected_validation_episode_ids=ids,results=results,complete=False,protocol='Initial-state recreation only; expert metrics stop at each source length, padding interactions count; BC runs full 200 steps; all diagnostic transitions excluded from learning.'),indent=2))
    report=json.loads((args.output/'diagnostic.json').read_text());report['complete']=True;(args.output/'diagnostic.json').write_text(json.dumps(report,indent=2))
    print(json.dumps({mode:dict(success_once=sum(e['success_once'] for e in r['episodes']),grasped_once=sum(e['is_grasped_once'] for e in r['episodes']),placed_once=sum(e['is_obj_placed_once'] for e in r['episodes']),initial_obs_error=r['initial_observation_max_abs_error']) for mode,r in results.items()},indent=2))


if __name__=='__main__': main()
