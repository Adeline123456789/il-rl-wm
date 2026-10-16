"""CPU-only held-out action errors of a fixed BC checkpoint; no rollout score selection."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import h5py
import numpy as np
import torch
from scripts.train_bc import Actor


def main():
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--audit',type=Path,required=True);p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    torch.set_num_threads(2)
    split=json.loads(args.audit.read_text());ckpt=torch.load(args.checkpoint,map_location='cpu');assert split['alignment_validated']
    with h5py.File(split['dataset']) as f:
        first=f[f"traj_{split['validation_episode_ids'][0]}"]
        policy=Actor(first['obs'].shape[1],first['actions'].shape[1]);policy.load_state_dict(ckpt['actor']);policy.eval();actual=[];prediction=[];episodes=[]
        with torch.no_grad():
            for i in split['validation_episode_ids']:
                data=f[f'traj_{i}'];target=data['actions'][:];pred=policy(torch.as_tensor(data['obs'][:-1],dtype=torch.float32)).numpy();actual.append(target);prediction.append(pred);episodes.append(dict(episode_id=i,transitions=len(target),mse=float(np.mean((pred-target)**2))))
    target=np.concatenate(actual);pred=np.concatenate(prediction);error=pred-target
    result=dict(checkpoint=str(args.checkpoint),checkpoint_sha256=hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),dataset_sha256=split['dataset_sha256'],training_iteration=ckpt['iteration'],validation_episode_count=len(episodes),episodes=episodes,action_mse=float(np.mean(error**2)),mse_by_action_dimension=np.mean(error**2,axis=0).tolist(),mae_by_action_dimension=np.mean(np.abs(error),axis=0).tolist(),p95_absolute_error_by_action_dimension=np.quantile(np.abs(error),.95,axis=0).tolist(),expert_action_mean=target.mean(0).tolist(),predicted_action_mean=pred.mean(0).tolist(),expert_near_bound_fraction=(np.abs(target)>.95).mean(0).tolist(),predicted_near_bound_fraction=(np.abs(pred)>.95).mean(0).tolist(),interpretation='Held-out expert-state action accuracy only; cannot establish closed-loop success or policy robustness.')
    args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(result,indent=2));print(json.dumps({k:v for k,v in result.items() if k!='episodes'}))


if __name__=='__main__': main()
