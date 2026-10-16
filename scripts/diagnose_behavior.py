"""Describe actual PickCube action sources and behavior; no causal attribution."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import torch

SOURCES={0:'IL',1:'planned_SAC',2:'random_SAC',3:'bare_SAC'}


def extract(path):
    state=torch.load(path,map_location='cpu');r=state['replay']
    assert state['config']['env_id']=='PickCube-v1'
    assert state['global_step']==262144 and r['full'] and r['pos']==0
    obs=r['obs'].flatten(0,1).numpy();nxt=r['next_obs'].flatten(0,1).numpy()
    assert obs.shape==(262144,42)
    # qpos9/qvel9/grasp1/tcp7/goal3/object7/tcp-to-object3/object-to-goal3.
    for x in [obs,nxt]:
        np.testing.assert_allclose(x[:,36:39],x[:,29:32]-x[:,19:22],atol=1e-5,rtol=0)
        np.testing.assert_allclose(x[:,39:42],x[:,26:29]-x[:,29:32],atol=1e-5,rtol=0)
    data=dict(source=r['action_sources'].flatten().numpy(),reward=r['rewards'].flatten().numpy(),
        episode_id=r['episode_ids'].flatten().numpy(),timestep=r['timesteps'].flatten().numpy(),
        action=r['actions'].flatten(0,1).numpy(),grasped=obs[:,18]>.5,
        object_displacement=np.linalg.norm(nxt[:,29:32]-obs[:,29:32],axis=-1),
        next_goal_distance=np.linalg.norm(nxt[:,39:42],axis=-1))
    assert len(data['source'])==state['global_step']
    assert np.isin(data['source'],list(SOURCES)).all()
    return data,dict(checkpoint=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                     training_seed=state['config']['seed'],inherited_interactions=state['lineage_training_interactions']-state['global_step'])


def summarize(data,bin_size=65536):
    source=data['source'];n=len(source);assert n==262144
    assert bin_size in [65536,8192]
    rows=[]
    for start in range(0,n,bin_size):
        for code,label in SOURCES.items():
            idx=np.arange(start,start+bin_size);idx=idx[source[idx]==code]
            if not len(idx):continue
            rows.append(dict(start=start,end=start+bin_size,source=label,executions=len(idx),
                execution_fraction=len(idx)/bin_size,positive_reward_fraction=float((data['reward'][idx]>0).mean()),
                grasped_fraction=float(data['grasped'][idx].mean()),
                action_std=data['action'][idx].std(axis=0).tolist(),
                object_displacement_mean_m=float(data['object_displacement'][idx].mean()),
                object_displacement_p95_m=float(np.quantile(data['object_displacement'][idx],.95)),
                next_goal_distance_mean_m=float(data['next_goal_distance'][idx].mean())))
    success=np.flatnonzero(data['reward']>0)
    _,first=np.unique(data['episode_id'][success],return_index=True)
    first=success[first]
    return dict(new_environment_interactions=0,recorded_real_transitions=n,bins=rows,
        successful_episodes_in_segment=len(first),episodes_seen=len(np.unique(data['episode_id'])),
        first_success_action_source={label:int((source[first]==code).sum()) for code,label in SOURCES.items()},
        limitations='Observed source association, not causal action credit or matched-state comparisons. Segment includes partial boundary episodes. Sustained reward fraction is not episode success rate. Action std is observed dispersion, not policy entropy or evidence of useful exploration.')


def main():
    p=argparse.ArgumentParser();group=p.add_mutually_exclusive_group(required=True)
    group.add_argument('--checkpoint',type=Path);group.add_argument('--input-arrays',type=Path)
    p.add_argument('--bin-size',type=int,choices=[65536,8192],default=65536)
    p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    args.output.mkdir(parents=True,exist_ok=False);torch.set_num_threads(2)
    if args.checkpoint:
        data,provenance=extract(args.checkpoint)
        np.savez_compressed(args.output/'transitions.npz',**data)
        (args.output/'source.json').write_text(json.dumps(provenance,indent=2)+'\n')
    else:
        with np.load(args.input_arrays) as f:data={key:f[key] for key in f.files}
    result=summarize(data,args.bin_size);(args.output/'result.json').write_text(json.dumps(result,indent=2)+'\n')


if __name__=='__main__':main()
