import unittest
import numpy as np
import torch
from scripts.world_model import WorldModel
from scripts.wm_data import append_episode, packed


class WorldModelTest(unittest.TestCase):
    def test_two_step_recursion_and_gradient(self):
        # A linear dynamics oracle isolates recurrence from network fitting.
        model=WorldModel(torch.zeros(1),torch.ones(1),1)
        model.net=torch.nn.Linear(2,2,bias=False)
        with torch.no_grad():model.net.weight.copy_(torch.tensor([[1.,1.],[0.,1.]]))
        states,_=model.rollout(torch.tensor([[1.]]),torch.tensor([[[2.],[3.]]]))
        torch.testing.assert_close(states,torch.tensor([[[4.],[11.]]]))
        states[0,1,0].backward()
        # ds2/dw_state includes the first step through the recursive prediction.
        self.assertAlmostEqual(float(model.net.weight.grad[0,0]),6.)

    def test_real_sequence_alignment_and_failure_stratum(self):
        obs=np.arange(4,dtype=np.float32).reshape(-1,1);rows=[]
        append_episode(rows,obs[:-1],obs[1:],np.zeros((3,1),np.float32),np.array([0.,0.,1.],np.float32),np.arange(3),True)
        data=packed(rows)
        torch.testing.assert_close(data['states'][:,:,0],torch.tensor([[0.,1.,2.],[1.,2.,3.]]))
        self.assertEqual(data['pre_success'].sum(),3)
        self.assertFalse(data['failure'].any())
        with self.assertRaises(AssertionError):
            append_episode([],obs[:-1],obs[1:]+1,np.zeros((3,1)),np.zeros(3),np.arange(3),True)


if __name__=='__main__':unittest.main()
