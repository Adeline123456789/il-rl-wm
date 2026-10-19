"""Frozen ACT/MLP-SAC/Transformer-SAC evaluation, with an explicit final-test gate."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from scripts.act_inference import action_callable,load_model
from scripts.interaction_counter import InteractionCounter
from scripts.train_act_teacher import closed_loop


def validate_protocol(kind, mode, seed, episodes, num_envs, gate):
    assert kind in ('teacher','mlp','act')
    assert mode in (('first','ensemble') if kind=='teacher' else ('mean','stochastic'))
    assert seed in (10000,20000) and num_envs in (8,10)
    assert episodes>0 and episodes%num_envs==0
    if seed==20000:
        assert episodes==100 and num_envs==10
        assert json.loads(Path(gate).read_text())['released'], 'Final-test gate closed'


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--checkpoint',type=Path,required=True)
    parser.add_argument('--kind',choices=['teacher','mlp','act'],required=True)
    parser.add_argument('--mode',choices=['first','ensemble','mean','stochastic'],required=True)
    parser.add_argument('--seed',type=int,choices=[10000,20000],required=True)
    parser.add_argument('--episodes',type=int,required=True)
    parser.add_argument('--num-envs',type=int,default=10)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--training-seed',type=int,default=0)
    parser.add_argument('--training-interactions',type=int,default=0)
    args=parser.parse_args()
    validate_protocol(args.kind,args.mode,args.seed,args.episodes,args.num_envs,'configs/act_final_test_gate.json')
    torch.set_num_threads(2);torch.manual_seed(args.seed)
    saved=torch.load(args.checkpoint,map_location='cpu',weights_only=False)
    if args.training_interactions:
        assert saved['training_interactions']==args.training_interactions
        assert saved['config']['seed']==args.training_seed
    model=load_model(saved,args.kind,'cuda')
    args.output.mkdir(parents=True,exist_ok=True)
    assert not any((args.output/name).exists() for name in ['result.json','episodes.jsonl','validation_episodes.jsonl'])
    purpose='final_test' if args.seed==20000 else 'validation'
    counter=InteractionCounter(args.output/'interactions.json')
    result=closed_loop(action_callable(model,args.kind,args.mode),args.output,saved.get('step',0),
        args.episodes,counter,temporal=args.mode=='ensemble',method=args.kind+'_'+args.mode,
        eval_seed=args.seed,num_envs=args.num_envs,purpose=purpose,trace=True)
    result.update(seed=args.seed,training_seed=args.training_seed,training_interactions=args.training_interactions,
        checkpoint=str(args.checkpoint),checkpoint_sha256=hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
        kind=args.kind,mode=args.mode,purpose=purpose,num_envs=args.num_envs,
        new_training_interactions=0,interactions=args.episodes*200,selection='explicit fixed checkpoint',
        protocol=dict(env_id='PickCube-v1',obs_mode='state',control_mode='pd_ee_delta_pos',reward_mode='sparse',
                      horizon=200,max_episode_steps=200,reset_protocol='explicit_batch_seed'),
        torch_peak_allocated_bytes=torch.cuda.max_memory_allocated())
    (args.output/'result.json').write_text(json.dumps(result,indent=2)+'\n')


if __name__=='__main__':main()
