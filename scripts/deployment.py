"""Standalone SAC + H2 world-model planner. This package has no IL weights/calls."""
import torch
from torch import nn
from scripts.load_components import sac_actor,world_model,task_value
from scripts.planning import generate_candidates,score_candidates


class DeploymentPolicy(nn.Module):
    def __init__(self, package, device='cpu'):
        super().__init__()
        assert set(package)=={'actor','wm','value','protocol','source'}
        assert package['protocol']['sac_candidates']==9 and package['protocol']['horizon']==2
        assert package['protocol']['gamma']==.99
        self.actor=sac_actor(package,device)
        self.wm=world_model({'model':package['wm']},device)
        self.value=task_value({'value':package['value']},device)
        self.requires_grad_(False)
        self.eval()

    @torch.no_grad()
    def get_eval_action(self, observation):
        candidates=generate_candidates(observation,self.actor,self.wm,il=None)
        return score_candidates(candidates,self.actor,self.value)['action']
