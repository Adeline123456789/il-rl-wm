import unittest
import torch
from scripts.u0_modules import U0


class ConstantStudent:
    def get_eval_action(self,s):return torch.full_like(s,.1)
    def get_action(self,s):return torch.full_like(s,.1),None,None


class U0RoutingTest(unittest.TestCase):
    def test_exploration_changes_executed_action_and_counts_only_real_choices(self):
        module=object.__new__(U0)
        module.actor=ConstantStudent();module.il=lambda s:torch.full_like(s,.8)
        module.wm=lambda s,a:(s+a,torch.zeros(len(s)))
        module.value=lambda s,a:s[:,0]
        module.counts=torch.zeros(3,dtype=torch.int64);module.chosen=torch.zeros(9,dtype=torch.int64)
        torch.manual_seed(31);expected=torch.rand(100)<.2
        torch.manual_seed(31);action,source,selected=module.act(torch.zeros(100,1),step=1)
        self.assertTrue((selected==0).all())
        torch.testing.assert_close(action[:,0],torch.where(expected,.1,.8))
        self.assertTrue(torch.equal(source==2,expected))
        self.assertEqual(module.counts.tolist(),[int((~expected).sum()),0,int(expected.sum())])
        self.assertEqual(int(module.counts.sum()),100) # 900 candidate proposals are not executions.


if __name__=='__main__':unittest.main()
