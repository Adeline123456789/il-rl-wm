import unittest
import torch
from scripts.planning import generate_candidates, discounted_score, score_candidates
from scripts.task_value import TaskValue, make_target, update_value


class Policy:
    def get_eval_action(self,state):return .1*state
    def get_action(self,state):return .1*state+.01*torch.randn_like(state),None,None


class PlanningTest(unittest.TestCase):
    def test_recursive_policy_inputs_and_candidate_budget(self):
        def wm(s,a):return s+a,torch.zeros(len(s))
        obs=torch.ones(3,1)
        il=lambda s:.2*s
        candidate=generate_candidates(obs,Policy(),wm,il,deterministic=True)
        self.assertEqual(candidate['actions'].shape,(3,9,2,1))
        torch.testing.assert_close(candidate['actions'][:,0,:,0],torch.tensor([[.2,.24]]).expand(3,2))
        torch.testing.assert_close(candidate['actions'][:,1,:,0],torch.tensor([[.1,.11]]).expand(3,2))
        torch.testing.assert_close(candidate['states'][:,:,0],obs[:,None,:].expand(3,9,1))
        no_il=generate_candidates(obs,Policy(),wm)
        self.assertEqual(no_il['sources'],['sac']*9)
        self.assertGreater(float(no_il['actions'][0,:,0,0].std()),0.)

    def test_hand_computed_score_and_actual_first_action(self):
        rewards=torch.tensor([[[1.,2.],[0.,0.]]]);values=torch.tensor([[3.,3.]])
        score,short,tail=discounted_score(rewards,values,.9)
        torch.testing.assert_close(score,torch.tensor([[5.23,2.43]]))
        torch.testing.assert_close(short,torch.tensor([[2.8,0.]]))
        torch.testing.assert_close(tail,torch.tensor([[2.43,2.43]]))
        candidates=dict(rewards=rewards,states=torch.zeros(1,2,3,1),actions=torch.tensor([[[[.2],[.3]],[[.7],[.8]]]]))
        result=score_candidates(candidates,Policy(),lambda s,a:torch.full((len(s),),3.),.9)
        torch.testing.assert_close(result['action'],torch.tensor([[.2]]))

    def test_task_td_does_not_update_actor_or_add_entropy(self):
        actor=torch.nn.Linear(1,1,bias=False)
        actor.get_eval_action=actor.forward
        value=TaskValue(torch.zeros(1),torch.ones(1),1)
        value.net=torch.nn.Linear(2,1)
        with torch.no_grad():value.net.weight.zero_();value.net.bias.fill_(2.)
        target=make_target(value);before=actor.weight.detach().clone()
        optimizer=torch.optim.SGD(value.parameters(),lr=0.)
        loss=update_value(value,target,optimizer,actor,torch.ones(2,1),torch.zeros(2,1),torch.ones(2),torch.ones(2,1),gamma=.5)
        self.assertEqual(float(loss),0.) # 1 + .5*2 = 2; prediction=2
        self.assertIsNone(actor.weight.grad)
        torch.testing.assert_close(actor.weight,before)


if __name__=='__main__':unittest.main()
