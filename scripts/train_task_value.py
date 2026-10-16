"""Preheat the additional task evaluator for one frozen deterministic student."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from scripts.load_components import sac_actor,world_model
from scripts.task_value import TaskValue,make_target,update_value
from scripts.wm_data import load_sequences


@torch.no_grad()
def evaluate(value,target,actor,data):
    result={}
    for source in ['expert','online']:
        rows=data[source+'_validation'];predictions=[];residuals=[]
        for i in range(0,len(rows['actions']),1024):
            states=rows['states'][i:i+1024];actions=rows['actions'][i:i+1024,0];reward=rows['rewards'][i:i+1024,0]
            q=value(states[:,0],actions);y=reward+.99*target(states[:,1],actor.get_eval_action(states[:,1]))
            predictions.append(q.cpu());residuals.append((q-y).cpu())
        q=torch.cat(predictions);residual=torch.cat(residuals)
        result[source]=dict(samples=len(q),td_mse=float(residual.square().mean()),td_mae=float(residual.abs().mean()),q_mean=float(q.mean()),
            q_min=float(q.min()),q_max=float(q.max()),outside_known_return_range_fraction=float(((q<0)|(q>100)).float().mean()),
            caution='Held-out Bellman consistency, not verified Monte Carlo value accuracy')
    return result


def main():
    p=argparse.ArgumentParser();p.add_argument('--expert-audit',type=Path,required=True);p.add_argument('--sac',type=Path,required=True);p.add_argument('--wm',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--updates',type=int,default=5000);p.add_argument('--seed',type=int,default=0);args=p.parse_args()
    args.output.mkdir(exist_ok=True,parents=True);assert not (args.output/'train.jsonl').exists()
    torch.set_num_threads(4);torch.manual_seed(args.seed);np.random.seed(args.seed)
    data,manifest=load_sequences(args.expert_audit,args.sac)
    data={k:{name:v.cuda() for name,v in rows.items()} for k,rows in data.items()}
    actor=sac_actor(torch.load(args.sac,map_location='cpu'),'cuda').eval();actor.requires_grad_(False)
    wm=world_model(torch.load(args.wm,map_location='cpu'),'cuda')
    value=TaskValue(wm.obs_mean,wm.obs_std,actor.fc_mean.out_features).cuda();target=make_target(value)
    optimizer=torch.optim.Adam(value.parameters(),lr=3e-4);start=time.monotonic()
    config=dict(updates=args.updates,batch_size=512,seed=args.seed,gamma=.99,tau=.01,lr=3e-4,expert_fraction=.5,
        continuation='frozen deterministic SAC mean',sequence_sampling='first transition of each H2 sequence; excludes final transition per episode for preheat only',
        sac=str(args.sac),wm=str(args.wm),wm_sha256=hashlib.sha256(args.wm.read_bytes()).hexdigest(),data=manifest)
    (args.output/'config.json').write_text(json.dumps(config,indent=2))
    (args.output/'initial_validation.json').write_text(json.dumps(evaluate(value,target,actor,data),indent=2))
    for update in range(1,args.updates+1):
        samples=[]
        for source in ['expert','online']:
            rows=data[source+'_train'];ix=torch.randint(len(rows['actions']),(256,),device='cuda')
            samples.append(dict(obs=rows['states'][ix,0],next_obs=rows['states'][ix,1],actions=rows['actions'][ix,0],rewards=rows['rewards'][ix,0]))
        batch={k:torch.cat([s[k] for s in samples]) for k in samples[0]}
        loss=update_value(value,target,optimizer,actor,**batch);assert torch.isfinite(loss)
        if update==1 or update%100==0:
            with (args.output/'train.jsonl').open('a') as f:f.write(json.dumps(dict(update=update,td_loss=float(loss)))+'\n')
    (args.output/'validation.json').write_text(json.dumps(evaluate(value,target,actor,data),indent=2))
    directory=Path('checkpoints')/args.output.name;directory.mkdir(exist_ok=True,parents=True);path=directory/'final.pt'
    torch.save(dict(value=value.state_dict(),target=target.state_dict(),optimizer=optimizer.state_dict(),config=config,updates=args.updates,
        rng=dict(torch=torch.get_rng_state(),cuda=torch.cuda.get_rng_state_all(),numpy=np.random.get_state())),path)
    (args.output/'result.json').write_text(json.dumps(dict(updates=args.updates,new_environment_interactions=0,elapsed_seconds=time.monotonic()-start,
        trainable_parameters=sum(p.numel() for p in value.parameters()),checkpoint=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),torch_peak_allocated_bytes=torch.cuda.max_memory_allocated()),indent=2))


if __name__=='__main__':main()
