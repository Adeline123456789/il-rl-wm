"""Two-step real sequences, split by complete episode identity, never across resets."""
import hashlib
import json
from pathlib import Path
import h5py
import numpy as np
import torch


def append_episode(rows, obs, next_obs, actions, rewards, timesteps, complete):
    n=len(actions)
    if n<2:return
    assert np.array_equal(next_obs[:-1],obs[1:]), 'Real next observation discontinuity'
    assert np.all(np.diff(timesteps)==1)
    first_success=np.flatnonzero(rewards>0)
    pre=np.zeros(n,dtype=bool)
    if len(first_success):pre[max(0,first_success[0]-5):first_success[0]]=True
    rows.append(dict(states=np.stack([obs[:-1],next_obs[:-1],next_obs[1:]],1),
                     actions=np.stack([actions[:-1],actions[1:]],1),
                     rewards=np.stack([rewards[:-1],rewards[1:]],1),
                     pre_success=np.stack([pre[:-1],pre[1:]],1),
                     failure=np.full((n-1,2),complete and not len(first_success),dtype=bool)))


def packed(rows):
    assert rows, 'Need real sequences in each split'
    return {k:torch.from_numpy(np.concatenate([r[k] for r in rows])).float() for k in rows[0]}


def load_sequences(audit_path, checkpoint_path):
    audit_path=Path(audit_path);checkpoint_path=Path(checkpoint_path)
    audit=json.loads(audit_path.read_text())
    assert audit['alignment_validated']
    assert not (audit_path.parent/'INITIAL_OBSERVATION_CAVEAT.json').exists()
    assert hashlib.sha256(Path(audit['dataset']).read_bytes()).hexdigest()==audit['dataset_sha256']
    data={};manifest=dict(expert_audit=str(audit_path),expert_dataset_sha256=audit['dataset_sha256'],online_checkpoint=str(checkpoint_path),online_checkpoint_sha256=hashlib.sha256(checkpoint_path.read_bytes()).hexdigest(),splits={})
    with h5py.File(audit['dataset']) as f:
        for split in ['train','validation']:
            rows=[];ids=audit[split+'_episode_ids']
            for i in ids:
                e=f[f'traj_{i}'];obs=e['obs'][:];n=len(e['actions'])
                append_episode(rows,obs[:-1],obs[1:],e['actions'][:],e['rewards'][:],np.arange(n),True)
            data['expert_'+split]=packed(rows);manifest['splits']['expert_'+split]=ids
    snapshot=torch.load(checkpoint_path,map_location='cpu');r=snapshot['replay']
    assert snapshot['config']['reconfiguration_freq']==1, 'Use corrected online reset protocol'
    assert snapshot['config']['env_id']==audit['env_info']['env_id']
    assert snapshot['config']['control_mode']==audit['env_info']['env_kwargs']['control_mode']
    # Sort each saved episode by timestep, independent of the replay ring position.
    columns={k:r[k].flatten(0,1).numpy() for k in ['obs','next_obs','actions','rewards','episode_ids','timesteps','raw_truncated']}
    if snapshot['config']['env_id']=='PickCube-v1':
        initial=columns['obs'][columns['timesteps']==0]
        stale=(initial[:,18]>.5)&(np.linalg.norm(initial[:,36:39],axis=1)>.1)
        manifest['online_initial_contact_audit']=dict(count=len(initial),stale_count=int(stale.sum()))
        assert not stale.any(), 'F-E0-004 persists in online data'
    ids=np.unique(columns['episode_ids']);order=np.random.default_rng(42).permutation(ids)
    cut=max(1,int(.8*len(order)));assert cut<len(order)
    for split,selected in [('train',order[:cut]),('validation',order[cut:])]:
        rows=[]
        for i in selected:
            ix=np.flatnonzero(columns['episode_ids']==i);ix=ix[np.argsort(columns['timesteps'][ix])]
            complete=columns['timesteps'][ix[0]]==0 and bool(columns['raw_truncated'][ix[-1]])
            append_episode(rows,*(columns[k][ix] for k in ['obs','next_obs','actions','rewards','timesteps']),complete)
        data['online_'+split]=packed(rows);manifest['splits']['online_'+split]=selected.tolist()
    for k,v in data.items():
        assert torch.isfinite(v['states']).all() and torch.isfinite(v['actions']).all()
        assert ((v['rewards']==0)|(v['rewards']==1)).all()
    manifest['sequence_counts']={k:len(v['actions']) for k,v in data.items()}
    manifest['definition']='H=2 consecutive real transitions; SAC episode split seed42; existing expert split; no imagined transitions; failures require complete recorded episode'
    return data,manifest
