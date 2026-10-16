"""Fixed real-input intervention probe; not training or a performance comparison."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from torch import nn
from scripts.train_sac import SoftQNetwork


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--checkpoint',type=Path,required=True)
    parser.add_argument('--inputs',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=False);torch.set_num_threads(2)
    checkpoint=torch.load(args.checkpoint,map_location='cpu')
    assert not checkpoint['config'].get('critic_layer_norm',False)
    with np.load(args.inputs) as data:
        obs=torch.from_numpy(np.concatenate([data[s+'_obs'] for s in ['expert','online']]))
        action=torch.from_numpy(np.concatenate([data[s+'_actions'] for s in ['expert','online']]))
    env=SimpleNamespace(single_observation_space=SimpleNamespace(shape=(obs.shape[-1],)),
                        single_action_space=SimpleNamespace(shape=(action.shape[-1],)))
    arrays={};results={}
    for key in ['qf1','qf2']:
        plain=SoftQNetwork(env);plain.load_state_dict(checkpoint[key])
        normalized=SoftQNetwork(env,layer_norm=True)
        for old,new in zip([m for m in plain.modules() if isinstance(m,nn.Linear)],
                           [m for m in normalized.modules() if isinstance(m,nn.Linear)]):
            new.load_state_dict(old.state_dict());assert torch.equal(new.weight,old.weight)
        for name,model in [('original',plain),('inserted_norm',normalized)]:
            model.eval();model.requires_grad_(False)
            inputs=action.clone().requires_grad_(True)
            prediction=model(obs,inputs).squeeze(-1)
            gradient=torch.autograd.grad(prediction.sum(),inputs)[0].norm(dim=-1)
            assert torch.isfinite(prediction).all() and torch.isfinite(gradient).all()
            arrays[key+'_'+name+'_q']=prediction.detach().numpy()
            arrays[key+'_'+name+'_action_gradient_norm']=gradient.numpy()
            results[key+'_'+name]=dict(q_mean=float(prediction.mean()),q_std=float(prediction.std()),
                mean_action_gradient_norm=float(gradient.mean()),max_action_gradient_norm=float(gradient.max()))
    np.savez_compressed(args.output/'values.npz',**arrays)
    report=dict(checkpoint=str(args.checkpoint),checkpoint_sha256=hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
        inputs=str(args.inputs),inputs_sha256=hashlib.sha256(args.inputs.read_bytes()).hexdigest(),
        samples=len(obs),new_environment_interactions=0,new_gradient_updates=0,results=results,
        interpretation='Direct insertion changes the already learned critic function. This is not a calibrated estimate or evidence of better behavior. U1 must use the registered matched-initialization full training comparison, not silently replace a trained critic.')
    (args.output/'result.json').write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':main()
