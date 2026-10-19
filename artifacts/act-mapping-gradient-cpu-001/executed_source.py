"""Read-only CPU derivatives on registered real expert-train states; no optimizer."""
import argparse, hashlib, json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from scripts.act_inference import load_model
from scripts.diagnose_act_online_drift import train_probe, digest

def fingerprint(model):
    h=hashlib.sha256()
    for k,v in model.state_dict().items():
        h.update(k.encode()); h.update(v.detach().cpu().numpy().tobytes())
    return h.hexdigest()

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=False);torch.set_num_threads(2)
    obs,ids,times,audit=train_probe(Path('artifacts/e0-pickcube-gpu-replay-full-003/dataset-audit.json'))
    init_path=Path('releases/act-c1-init-001/weights.pt')
    initial=torch.load(init_path,map_location='cpu',weights_only=False)
    actor=load_model(initial,'act').requires_grad_(True)
    arrays=dict(observations=obs.numpy(),episode_ids=ids,timestep=times);results=[]
    for label,folder in [('T5','act-t5-online-seed0-001'),('U2','act-u2-retention-seed0-001')]:
        records=[json.loads(s) for s in (Path('artifacts')/folder/'checkpoints.jsonl').read_text().splitlines()]
        for step in [0,16384,65536,131072,262144]:
            source=dict(path=str(init_path),sha256=digest(init_path)) if step==0 else next(r for r in records if r['step']==step)
            assert digest(source['path'])==source['sha256']
            state=torch.load(source['path'],map_location='cpu',weights_only=False)
            actor.load_state_dict(state['actor']);del state
            before=fingerprint(actor);parts={k:[] for k in ['raw','mean','action','log_std','dmean_draw','daction_draw','dsample_draw','dlogprob_draw']}
            norms=[];torch.manual_seed(719)
            for batch in obs.split(64):
                captured=[]
                hook=actor.action_head.register_forward_hook(lambda m,i,o:captured.append(o))
                mean,logs=actor(batch);raw=captured[-1];action=mean.tanh()
                dm=torch.autograd.grad(mean.sum(),raw,retain_graph=True)[0]
                da=torch.autograd.grad(action.sum(),raw,retain_graph=True)[0]
                params=list(actor.named_parameters())
                grads=torch.autograd.grad(action.mean(),[v for k,v in params],allow_unused=True)
                norms.append({group:float(torch.sqrt(sum((g.detach().square().sum() for (k,v),g in zip(params,grads) if g is not None and k.startswith(group)),torch.tensor(0.)))) for group in ['action_head','transformer','input_proj_robot_state','fc_logstd']})
                sample,lp,_=actor.get_action(batch);raw2=captured[-1]
                ds=torch.autograd.grad(sample.sum(),raw2,retain_graph=True)[0]
                dl=torch.autograd.grad(lp.sum(),raw2)[0];hook.remove()
                for k,v in dict(raw=raw,mean=mean,action=action,log_std=logs,dmean_draw=dm,daction_draw=da,dsample_draw=ds,dlogprob_draw=dl).items():
                    assert torch.isfinite(v).all(),(label,step,k)
                    parts[k].append(v.detach().numpy())
                assert torch.isfinite(lp).all()
            d={k:np.concatenate(v) for k,v in parts.items()};sat=np.abs(d['raw'])>1-actor.action_epsilon
            assert np.all(d['dmean_draw'][sat]==0) and np.all(d['daction_draw'][sat]==0)
            assert np.all(d['dsample_draw'][sat]==0) and np.all(d['dlogprob_draw'][sat]==0)
            np.testing.assert_allclose(d['daction_draw'][~sat],1,atol=2e-5,rtol=2e-5)
            assert fingerprint(actor)==before
            prefix=f'{label}_{step}'
            arrays.update({prefix+'_'+k:v for k,v in d.items()});arrays[prefix+'_saturated']=sat
            results.append(dict(method=label,step=step,checkpoint=source,model_unchanged=True,
                clipped_fraction_per_dim=sat.mean(0).tolist(),raw_abs_max=np.abs(d['raw']).max(0).tolist(),
                zero_action_derivative_fraction_per_dim=(d['daction_draw']==0).mean(0).tolist(),
                nonclipped_action_derivative_max_error=float(np.max(np.abs(d['daction_draw'][~sat]-1))),
                batch_mean_action_parameter_gradient_norms=norms))
    np.savez_compressed(a.output/'arrays.npz',**arrays)
    result=dict(results=results,fixed_real_training_states=len(obs),dataset_sha256=audit['dataset_sha256'],
        new_environment_interactions=0,gpu_used=False,optimizer_steps=0,final_test_states_used=False,
        interpretation='Derivatives through actual forward/get_action. Strictly clipped outputs block the action-mean path, including sampled-action and log-prob mean paths. Other action dimensions and shared log_std features can still update the backbone. Sum/mean VJPs are local sensitivity probes, not a SAC-loss causal attribution. Probe uses first 1024 train states and is not representative of all online states.')
    (a.output/'result.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps([{k:v for k,v in r.items() if k!='batch_mean_action_parameter_gradient_norms'} for r in results],indent=2))
if __name__=='__main__':main()
