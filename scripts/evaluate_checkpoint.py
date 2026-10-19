"""Frozen whole-episode evaluation, using the same helper as main SAC/U0."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
from types import SimpleNamespace
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import gymnasium as gym
import torch
import mani_skill.envs
from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv
from scripts.interaction_counter import InteractionCounter,CountedEnv
from scripts.load_components import sac_actor,world_model,task_value
from scripts.planning import generate_candidates,score_candidates
from scripts.sac_evidence import evaluate_policy
from scripts.u0_modules import InferencePolicy


def load_policies(checkpoint,methods,device):
    actor=sac_actor(checkpoint,device).eval();actor.requires_grad_(False)
    policies={'sac':actor}
    il=None
    if any(method in ['il','il_sac_wm'] for method in methods):
        from scripts.train_bc import Actor as BCActor
        il=BCActor(actor.backbone[0].in_features,actor.fc_mean.out_features).to(device)
        il.load_state_dict(checkpoint['frozen_il']);il.eval();il.requires_grad_(False)
        policies['il']=InferencePolicy(il)
    if any(method in ['sac_wm','il_sac_wm'] for method in methods):
        wm=world_model({'model':checkpoint['u0']['wm']},device).eval();wm.requires_grad_(False)
        value=task_value({'value':checkpoint['u0']['value']},device).eval();value.requires_grad_(False)
        for name,teacher in [('sac_wm',None),('il_sac_wm',il)]:
            policies[name]=InferencePolicy(lambda obs,teacher=teacher:score_candidates(
                generate_candidates(obs,actor,wm,teacher),actor,value)['action'])
    return {method:policies[method] for method in methods}


def main():
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--methods',nargs='+',choices=['il','sac','il_sac_wm','sac_wm'],required=True)
    p.add_argument('--seed',type=int,choices=[10000,20000],required=True)
    p.add_argument('--episodes',type=int,required=True);p.add_argument('--num-envs',type=int,default=10)
    args=p.parse_args();assert args.episodes>0 and args.episodes%args.num_envs==0
    assert len(args.methods)==len(set(args.methods))
    if args.seed==20000:assert args.episodes==100, 'Registered final test is exactly 100 episodes per model/method'
    torch.set_num_threads(4);checkpoint=torch.load(args.checkpoint,map_location='cpu')
    config=checkpoint['config'];assert config['gamma']==.99 and config['reconfiguration_freq']==1
    args.output.mkdir(exist_ok=True,parents=True);assert not (args.output/'episodes.jsonl').exists()
    policies=load_policies(checkpoint,args.methods,'cuda')
    purpose='final_test' if args.seed==20000 else 'validation'
    counter=InteractionCounter(args.output/'interactions.json')
    raw=gym.make(config['env_id'],num_envs=args.num_envs,obs_mode='state',sim_backend='physx_cuda',
        control_mode=config['control_mode'],reward_mode='sparse',reconfiguration_freq=1,
        max_episode_steps=config['max_episode_steps'])
    env=ManiSkillVectorEnv(CountedEnv(raw,counter,purpose),args.num_envs,ignore_terminations=True,record_metrics=True)
    eval_args=SimpleNamespace(eval_seed=args.seed,num_eval_envs=args.num_envs,max_episode_steps=config['max_episode_steps'],
        num_eval_steps=args.episodes//args.num_envs*config['max_episode_steps'],matched_eval_resets=True)
    record=dict(checkpoint=str(args.checkpoint),sha256=hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
        training_seed=config['seed'],lineage_training_interactions=checkpoint['lineage_training_interactions'],
        critic_layer_norm=config.get('critic_layer_norm',False),
        purpose=purpose,seed=args.seed,episodes_per_method=args.episodes,num_envs=args.num_envs,
        protocol={key:config[key] for key in ['env_id','control_mode','max_episode_steps','gamma','reconfiguration_freq']},
        checkpoint_selection='explicit fixed checkpoint; no selection by this evaluation',methods={},new_training_interactions=0)
    record['protocol']['reset_protocol']='explicit_batch_seed'
    for method,policy in policies.items():
        start=time.monotonic()
        interactions=evaluate_policy(policy,env,eval_args,args.output,checkpoint['global_step'],method,
                                     episode_filename='episodes.jsonl')
        record['methods'][method]=dict(interactions=interactions,wall_seconds=time.monotonic()-start,
            candidates=9 if method in ['sac_wm','il_sac_wm'] else 1,calls_il=method in ['il','il_sac_wm'])
    env.close();counter.flush()
    rows=[json.loads(line) for line in (args.output/'episodes.jsonl').read_text().splitlines()]
    for method in args.methods:
        episodes=[row for row in rows if row['method']==method];assert len(episodes)==args.episodes
        record['methods'][method].update(success_once=sum(row['success_once'] for row in episodes),
            success_at_end=sum(row['success_at_end'] for row in episodes),mean_return=sum(row['return'] for row in episodes)/len(episodes))
    (args.output/'result.json').write_text(json.dumps(record,indent=2)+'\n')


if __name__=='__main__':main()
