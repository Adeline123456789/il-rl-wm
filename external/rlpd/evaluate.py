"""Independent RLPD evaluation with the same whole-episode protocol as main SAC."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from types import SimpleNamespace
os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE','false')
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
import gymnasium as gym
import jax
import mani_skill.envs
from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv
import torch
from external.rlpd.policy import FrozenPolicy
from scripts.interaction_counter import CountedEnv,InteractionCounter
from scripts.sac_evidence import evaluate_policy


def main():
    p=argparse.ArgumentParser();p.add_argument('--policy',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--seed',type=int,choices=[10000,20000],required=True)
    p.add_argument('--episodes',type=int,required=True);p.add_argument('--num-envs',type=int,default=10)
    args=p.parse_args();assert args.episodes>0 and args.episodes%args.num_envs==0
    if args.seed==20000:assert args.episodes==100
    assert sys.prefix==str(ROOT/'.venv-rlpd')
    assert jax.devices()[0].platform=='gpu'
    torch.set_num_threads(4);policy=FrozenPolicy(args.policy);config=policy.manifest['config']
    assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT/'.cache/upstreams/rfcl',text=True).strip()==config['rfcl_commit']
    assert config['discount']==.99 and config['reward_mode']=='sparse'
    args.output.mkdir(parents=True,exist_ok=True);assert not (args.output/'episodes.jsonl').exists()
    purpose='final_test' if args.seed==20000 else 'validation'
    counter=InteractionCounter(args.output/'interactions.json')
    raw=gym.make(config['env_id'],num_envs=args.num_envs,obs_mode='state',sim_backend='physx_cuda',
        control_mode=config['control_mode'],reward_mode='sparse',reconfiguration_freq=1,
        max_episode_steps=config['max_episode_steps'])
    env=ManiSkillVectorEnv(CountedEnv(raw,counter,purpose),args.num_envs,ignore_terminations=True,record_metrics=True)
    eval_args=SimpleNamespace(eval_seed=args.seed,num_eval_envs=args.num_envs,
        max_episode_steps=config['max_episode_steps'],num_eval_steps=args.episodes//args.num_envs*config['max_episode_steps'])
    start=time.monotonic()
    interactions=evaluate_policy(policy,env,eval_args,args.output,policy.manifest['step'],'RLPD',episode_filename='episodes.jsonl')
    elapsed=time.monotonic()-start;env.close();counter.flush()
    rows=[json.loads(line) for line in (args.output/'episodes.jsonl').read_text().splitlines()]
    assert len(rows)==args.episodes
    record=dict(policy=str(args.policy),actor_sha256=policy.manifest['actor_sha256'],
        training_seed=config['seed'],training_interactions=policy.manifest['step'],
        purpose=purpose,seed=args.seed,episodes=args.episodes,num_envs=args.num_envs,
        success_once=sum(r['success_once'] for r in rows),success_at_end=sum(r['success_at_end'] for r in rows),
        mean_return=sum(r['return'] for r in rows)/len(rows),interactions=interactions,
        elapsed_seconds=elapsed,new_training_interactions=0,checkpoint_selection='fixed final',
        protocol=dict(env_id=config['env_id'],control_mode=config['control_mode'],max_episode_steps=config['max_episode_steps'],
                      reward_mode='sparse',gamma=.99,reconfiguration_freq=1,ignore_terminations=True))
    (args.output/'result.json').write_text(json.dumps(record,indent=2)+'\n')


if __name__=='__main__':main()
