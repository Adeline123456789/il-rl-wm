import unittest
from types import SimpleNamespace

import gymnasium as gym
import numpy as np
import torch

from scripts.train_bc import Actor as BCActor
from scripts.policy_bridge import freeze_il, fingerprint, initialize_sac_mean, check_frozen
from vendor.maniskill.sac.sac import Actor as SACActor


class PolicyBridgeTests(unittest.TestCase):
    def test_equal_means_independent_variance_and_frozen_teacher_after_update(self):
        torch.set_num_threads(2)
        torch.manual_seed(5)
        env = SimpleNamespace(single_observation_space=gym.spaces.Box(-np.inf,np.inf,(42,),dtype=np.float32),
                              single_action_space=gym.spaces.Box(-1,1,(4,),dtype=np.float32))
        bc, sac = BCActor(42,4), SACActor(env)
        frozen = freeze_il(bc)
        teacher_hash = fingerprint(frozen)
        variance_hash = fingerprint(sac.fc_logstd)
        initialize_sac_mean(bc,sac)
        observations = torch.from_numpy(np.load('artifacts/e0-pickcube-smoke-001/transitions.npz')['observation'][:4].reshape(-1,42))
        torch.testing.assert_close(bc(observations), sac.get_eval_action(observations), rtol=0, atol=0)
        self.assertEqual(variance_hash,fingerprint(sac.fc_logstd))
        sampled, logprob, _ = sac.get_action(observations)
        self.assertTrue(torch.isfinite(logprob).all())
        self.assertTrue(((sampled>=-1)&(sampled<=1)).all())
        self.assertFalse(torch.equal(sampled,sac.get_eval_action(observations)))
        optimizer = torch.optim.Adam(sac.parameters(),lr=1e-3)
        optimizer.zero_grad()
        (sampled.square().mean()+0.1*logprob.mean()).backward()
        optimizer.step()
        check_frozen(frozen,optimizer,teacher_hash)
        self.assertFalse(torch.equal(frozen(observations),sac.get_eval_action(observations)))


if __name__ == '__main__':
    unittest.main()
