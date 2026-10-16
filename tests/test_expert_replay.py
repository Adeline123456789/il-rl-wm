import hashlib
import json
from pathlib import Path
import tempfile
import unittest
import h5py
import numpy as np
import torch
from scripts.expert_replay import ExpertReplay


class ExpertReplayTests(unittest.TestCase):
    def test_heldout_episode_never_sampled_and_recording_end_bootstraps(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);dataset=root/'demo.h5'
            with h5py.File(dataset,'w') as f:
                for i in [0,1]:
                    g=f.create_group(f'traj_{i}');g['obs']=np.full((4,3),i*100,dtype=np.float32)+np.arange(4)[:,None];g['actions']=np.zeros((3,2));g['rewards']=np.array([0,0,1]);g['terminated']=np.array([0,0,1]);g['truncated']=np.array([0,0,1])
            audit=root/'audit.json';audit.write_text(json.dumps(dict(alignment_validated=True,env_info={'env_id':'PickCube-v1','env_kwargs':{'reward_mode':'sparse','control_mode':'pd_ee_delta_pos'}},dataset_sha256=hashlib.sha256(dataset.read_bytes()).hexdigest(),train_episode_ids=[0],validation_episode_ids=[1])))
            expert=ExpertReplay(dataset,audit,torch.device('cpu'));sample=expert.sample(128)
            self.assertEqual(expert.size,3)
            self.assertTrue((sample['obs']<3).all())
            torch.testing.assert_close(sample['next_obs'],sample['obs']+1)
            self.assertEqual(sample['dones'].sum(),0)
            self.assertGreater(sample['rewards'].sum(),0)
            sequence=expert.sample_sequences(128)
            self.assertTrue((sequence['states']<4).all())
            torch.testing.assert_close(sequence['states'][:,2],sequence['states'][:,0]+2)


if __name__=='__main__': unittest.main()
