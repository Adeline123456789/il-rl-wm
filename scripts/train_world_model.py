"""E4 side training on frozen real data; does not select or execute actions."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from scripts.world_model import WorldModel
from scripts.wm_data import load_sequences


@torch.no_grad()
def evaluate(model, data):
    result={}
    for source in ['expert','online']:
        rows=data[source+'_validation'];errors=[];probs=[]
        for i in range(0,len(rows['actions']),1024):
            states,logits=model.rollout(rows['states'][i:i+1024,0],rows['actions'][i:i+1024])
            errors.append(((states-rows['states'][i:i+1024,1:])/model.obs_std).square().mean(-1).cpu())
            probs.append(logits.sigmoid().cpu())
        error=torch.cat(errors);prob=torch.cat(probs);rewards=rows['rewards'].cpu()
        strata={'all':torch.ones_like(rewards,dtype=torch.bool),'zero':rewards==0,'nonzero':rewards>0,
                'pre_success_5':rows['pre_success'].cpu().bool(),'failed_episode':rows['failure'].cpu().bool()}
        result[source]={}
        for h in range(2):
            result[source][str(h+1)]={}
            for name,mask in strata.items():
                keep=mask[:,h];n=int(keep.sum());p=prob[keep,h];r=rewards[keep,h];pred=p>=.5;actual=r>0
                result[source][str(h+1)][name]=dict(count=n,state_normalized_mse=float(error[keep,h].mean()) if n else None,
                    reward_brier=float((p-r).square().mean()) if n else None,
                    true_positive=int((pred&actual).sum()),false_positive=int((pred&~actual).sum()),
                    false_negative=int((~pred&actual).sum()),true_negative=int((~pred&~actual).sum()))
    return result


def main():
    p=argparse.ArgumentParser();p.add_argument('--expert-audit',type=Path,required=True);p.add_argument('--online-checkpoint',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--updates',type=int,default=10000);p.add_argument('--batch-size',type=int,default=512);p.add_argument('--seed',type=int,default=0);args=p.parse_args()
    torch.set_num_threads(4);torch.manual_seed(args.seed);np.random.seed(args.seed)
    args.output.mkdir(exist_ok=True,parents=True);assert not (args.output/'train.jsonl').exists();assert args.batch_size%2==0
    (args.output/'config.json').write_text(json.dumps({k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},indent=2))
    data,manifest=load_sequences(args.expert_audit,args.online_checkpoint)
    (args.output/'data.json').write_text(json.dumps(manifest,indent=2))
    start_states=torch.cat([data[s+'_train']['states'][:,0] for s in ['expert','online']])
    model=WorldModel(start_states.mean(0),start_states.std(0),data['expert_train']['actions'].shape[-1]).cuda()
    data={k:{name:value.cuda() for name,value in rows.items()} for k,rows in data.items()}
    optimizer=torch.optim.Adam(model.parameters(),lr=3e-4);start=time.monotonic()
    (args.output/'initial_validation.json').write_text(json.dumps(evaluate(model,data),indent=2))
    for update in range(1,args.updates+1):
        batches=[]
        for source in ['expert','online']:
            rows=data[source+'_train'];ix=torch.randint(len(rows['actions']),(args.batch_size//2,),device='cuda')
            batches.append({k:v[ix] for k,v in rows.items()})
        batch={k:torch.cat([b[k] for b in batches]) for k in ['states','actions','rewards']}
        loss,components=model.loss(**batch);assert torch.isfinite(loss)
        optimizer.zero_grad();loss.backward();optimizer.step()
        if update%100==0 or update==1:
            with (args.output/'train.jsonl').open('a') as f:f.write(json.dumps(dict(update=update,loss=float(loss),**{k:float(v) for k,v in components.items()}))+'\n')
    validation=evaluate(model,data);(args.output/'validation.json').write_text(json.dumps(validation,indent=2))
    directory=Path('checkpoints')/args.output.name;directory.mkdir(exist_ok=True,parents=True);checkpoint=directory/'final.pt'
    torch.save(dict(model=model.state_dict(),optimizer=optimizer.state_dict(),updates=args.updates,seed=args.seed,data=manifest,rng=dict(torch=torch.get_rng_state(),cuda=torch.cuda.get_rng_state_all(),numpy=np.random.get_state())),checkpoint)
    (args.output/'result.json').write_text(json.dumps(dict(updates=args.updates,elapsed_seconds=time.monotonic()-start,trainable_parameters=sum(p.numel() for p in model.parameters()),new_environment_interactions=0,checkpoint=str(checkpoint),sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),torch_peak_allocated_bytes=torch.cuda.max_memory_allocated()),indent=2))


if __name__=='__main__':main()
