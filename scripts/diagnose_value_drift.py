"""Compare two U0 snapshots on identical real states; no simulator or learning."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from scripts.expert_replay import ExpertReplay
from scripts.load_components import sac_actor,task_value
from scripts.train_sac import SoftQNetwork


def stats(x):
    a=x.detach().numpy()
    return dict(mean=float(a.mean()),min=float(a.min()),max=float(a.max()),
                p05=float(np.quantile(a,.05)),p95=float(np.quantile(a,.95)))


def main():
    p=argparse.ArgumentParser();p.add_argument('--before',type=Path,required=True);p.add_argument('--after',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--fixed-batches',type=Path,help='Published real input samples; avoids loading the full replay/demonstrations')
    args=p.parse_args()
    args.output.mkdir(exist_ok=False,parents=True);torch.set_num_threads(2)
    before=torch.load(args.before,map_location='cpu');after=torch.load(args.after,map_location='cpu')
    keys=['obs','next_obs','actions','rewards']
    if args.fixed_batches:
        with np.load(args.fixed_batches) as saved:
            expert_indices=saved['expert_indices'];online_indices=saved['online_indices']
            batches={source:{k:torch.from_numpy(saved[f'{source}_{k}']) for k in keys} for source in ['expert','online']}
    else:
        expert=ExpertReplay(before['config']['expert_dataset'],before['config']['expert_audit'],'cpu')
        rng=np.random.default_rng(532)
        expert_indices=rng.integers(expert.size,size=2048)
        replay=before['replay'];online_count=replay['obs'].flatten(0,1).shape[0]
        online_indices=rng.integers(online_count,size=2048)
        batches={'expert':{k:expert.data[k][expert_indices] for k in keys},
                 'online':{k:replay[k].flatten(0,1)[online_indices] for k in keys}}
    np.savez_compressed(args.output/'fixed_batches.npz',expert_indices=expert_indices,online_indices=online_indices,
        **{f'{source}_{k}':v.numpy() for source,batch in batches.items() for k,v in batch.items()})
    report=dict(new_environment_interactions=0,reference_replay=str(args.before),sampling_seed=532,
                checkpoint_sha256={name:hashlib.sha256(path.read_bytes()).hexdigest() for name,path in [('before',args.before),('after',args.after)]},results={})
    report['fixed_batches_source']=str(args.fixed_batches) if args.fixed_batches else None
    report['original_training_sources']={name:snapshot.get('analysis_source') for name,snapshot in [('before',before),('after',after)]}
    arrays=dict(expert_indices=expert_indices,online_indices=online_indices)
    for label,snapshot in [('before',before),('after',after)]:
        actor=sac_actor(snapshot,'cpu').eval()
        env=SimpleNamespace(single_observation_space=SimpleNamespace(shape=(actor.backbone[0].in_features,)),single_action_space=SimpleNamespace(shape=(actor.fc_mean.out_features,)))
        critics=[]
        for key in ['qf1','qf2','qf1_target','qf2_target']:
            q=SoftQNetwork(env,layer_norm=snapshot['config'].get('critic_layer_norm',False));q.load_state_dict(snapshot[key]);q.eval();critics.append(q)
        q1,q2,qt1,qt2=critics
        value=task_value({'value':snapshot['u0']['value']},'cpu')
        target=task_value({'value':snapshot['u0']['value_target']},'cpu')
        alpha=snapshot['log_alpha'].exp().item();report['results'][label]={}
        with torch.no_grad():
            for source,batch in batches.items():
                obs,nxt,actions,reward=(batch[k] for k in ['obs','next_obs','actions','rewards'])
                mean=actor.get_eval_action(obs)
                torch.manual_seed(532);next_action,next_logp,_=actor.get_action(nxt)
                q_record=torch.minimum(q1(obs,actions),q2(obs,actions)).squeeze(-1)
                q_mean=torch.minimum(q1(obs,mean),q2(obs,mean)).squeeze(-1)
                soft_target=reward+.99*(torch.minimum(qt1(nxt,next_action),qt2(nxt,next_action))-alpha*next_logp).squeeze(-1)
                pure_record=value(obs,actions);pure_mean=value(obs,mean)
                pure_target=reward+.99*target(nxt,actor.get_eval_action(nxt))
                values=dict(soft_q_recorded=q_record,soft_q_mean_action=q_mean,soft_actor_recorded_gap=q_mean-q_record,
                    soft_td_residual=q_record-soft_target,soft_td_target=soft_target,next_sample_logp=next_logp.squeeze(-1),
                    pure_q_recorded=pure_record,pure_q_mean_action=pure_mean,pure_td_residual=pure_record-pure_target,
                    actor_recorded_action_mse=(mean-actions).square().mean(-1))
                report['results'][label][source]=dict(alpha=alpha,reward_fraction=float((reward>0).float().mean()),
                    negative_pure_mean_fraction=float((pure_mean<0).float().mean()),over100_pure_mean_fraction=float((pure_mean>100).float().mean()),
                    metrics={k:stats(v) for k,v in values.items()})
                arrays.update({f'{label}_{source}_{k}':v.numpy() for k,v in values.items()})
    report['limitations']='Fixed real-data evaluation and Bellman consistency, not Monte Carlo calibration or proof of causality. Soft SAC Q includes entropy; only the separate task value has the [0,100] continuing-reward bound. Same next-action Gaussian random seed, different actors. No held-out final test used.'
    np.savez_compressed(args.output/'values.npz',**arrays)
    (args.output/'result.json').write_text(json.dumps(report,indent=2))


if __name__=='__main__':main()
