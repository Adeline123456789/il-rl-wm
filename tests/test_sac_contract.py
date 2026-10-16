"""Numerical contracts tested independently of environment throughput/performance."""
import unittest
from types import SimpleNamespace
import numpy as np
import gymnasium as gym
import torch
from scripts.train_sac import Actor, ReplayBuffer


class SACContractTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        self.env = SimpleNamespace(single_observation_space=gym.spaces.Box(-np.inf,np.inf,(3,),dtype=np.float32),single_action_space=gym.spaces.Box(-2,4,(2,),dtype=np.float32))

    def test_scaled_tanh_density(self):
        torch.manual_seed(19)
        actor = Actor(self.env)
        obs = torch.randn(16,3)
        action, density, mean_action = actor.get_action(obs)
        mean, logstd = actor(obs)
        normalized = (action-actor.action_bias)/actor.action_scale
        pre_tanh = torch.atanh(normalized)
        normal_density = -0.5*((pre_tanh-mean)/logstd.exp()).square()-logstd-0.5*np.log(2*np.pi)
        jacobian = torch.log(actor.action_scale*(1-normalized.square())+1e-6)
        torch.testing.assert_close(density,(normal_density-jacobian).sum(1,keepdim=True),rtol=1e-4,atol=1e-5)
        torch.testing.assert_close(mean_action,torch.tanh(mean)*3+1)
        self.assertTrue(((action>=-2)&(action<=4)).all())

    def test_raw_labels_and_effective_mask_survive_ring_wrap(self):
        rb=ReplayBuffer(self.env,2,4,torch.device('cpu'),torch.device('cpu'))
        for t in range(3):
            rb.add(torch.ones(2,3)*t,torch.ones(2,3)*(t+1),torch.zeros(2,2),torch.ones(2),torch.zeros(2),torch.tensor([True,False]),torch.tensor([False,True]),torch.tensor([0,1]),torch.tensor([t,t]))
        self.assertTrue(rb.full)
        self.assertEqual(rb.pos,1)
        self.assertTrue(rb.raw_terminated[0,0])
        self.assertTrue(rb.raw_truncated[0,1])
        self.assertEqual(rb.timesteps[0,0],2)
        batch=rb.sample(64)
        self.assertEqual(batch.dones.sum(),0)
        torch.testing.assert_close(batch.next_obs,batch.obs+1)

    def test_two_step_sequences_wrap_ring_but_never_cross_reset(self):
        rb=ReplayBuffer(self.env,1,4,torch.device('cpu'),torch.device('cpu'))
        for t in range(6):
            episode=t//3;time_in_episode=t%3;value=100*episode+time_in_episode
            rb.add(torch.full((1,3),float(value)),torch.full((1,3),float(value+1)),torch.zeros(1,2),torch.zeros(1),torch.zeros(1),
                torch.tensor([False]),torch.tensor([time_in_episode==2]),torch.tensor([episode]),torch.tensor([time_in_episode]))
        sequence=rb.sample_sequences(1024)
        self.assertGreater(len(sequence['states']),0)
        self.assertEqual(set(sequence['states'][:,0,0].tolist()),{100.,101.})
        torch.testing.assert_close(sequence['states'][:,2],sequence['states'][:,0]+2)


if __name__ == '__main__': unittest.main()
