"""Fixed expert training-episode buffer for the declared SAC+50% demonstration baseline."""
import hashlib
import json
from pathlib import Path
import h5py
import numpy as np
import torch


class ExpertReplay:
    def __init__(self, dataset, audit, device):
        audit_path=Path(audit)
        assert not (audit_path.parent/'INVALIDATED.json').exists()
        assert not (audit_path.parent/'INITIAL_OBSERVATION_CAVEAT.json').exists()
        report=json.loads(audit_path.read_text())
        assert report['alignment_validated']
        assert report['env_info']['env_kwargs']['reward_mode']=='sparse'
        assert hashlib.sha256(Path(dataset).read_bytes()).hexdigest()==report['dataset_sha256']
        train=report['train_episode_ids'];assert not set(train)&set(report['validation_episode_ids'])
        columns={k:[] for k in ['obs','next_obs','actions','rewards','dones']}
        sequence_starts=[];offset=0
        with h5py.File(dataset) as source:
            for i in train:
                trajectory=source[f'traj_{i}'];n=len(trajectory['actions'])
                sequence_starts.extend(range(offset,offset+n-1));offset+=n
                columns['obs'].append(trajectory['obs'][:-1]);columns['next_obs'].append(trajectory['obs'][1:])
                columns['actions'].append(trajectory['actions'][:]);columns['rewards'].append(trajectory['rewards'][:])
                # Declared continuing-task protocol: raw labels stay in source HDF5.
                # Demo end is a recording boundary, not a task terminal state.
                columns['dones'].append(np.zeros(n,dtype=np.float32))
        self.data={k:torch.as_tensor(np.concatenate(v),dtype=torch.float32) for k,v in columns.items()}
        self.size=len(self.data['obs']);self.device=device
        self.sequence_starts=torch.tensor(sequence_starts,dtype=torch.int64)
        assert self.size>0
        self.manifest=dict(env_info=report['env_info'],dataset=str(dataset),dataset_sha256=report['dataset_sha256'],train_episode_ids=train,transitions=self.size,storage='cpu',effective_terminal_mask='always_bootstrap_continuing_task')

    def sample(self, n):
        indices=torch.randint(self.size,(n,))
        return {k:v[indices].to(self.device) for k,v in self.data.items()}

    def sample_sequences(self, n):
        i=self.sequence_starts[torch.randint(len(self.sequence_starts),(n,))]
        return dict(states=torch.stack([self.data['obs'][i],self.data['next_obs'][i],self.data['next_obs'][i+1]],1).to(self.device),
                    actions=torch.stack([self.data['actions'][i],self.data['actions'][i+1]],1).to(self.device),
                    rewards=torch.stack([self.data['rewards'][i],self.data['rewards'][i+1]],1).to(self.device))
