"""Extra task-return evaluator for the deterministic SAC mean continuation."""
import copy
import torch
from torch import nn


class TaskValue(nn.Module):
    def __init__(self, obs_mean, obs_std, action_dim):
        super().__init__()
        self.register_buffer('obs_mean',obs_mean.clone())
        self.register_buffer('obs_std',obs_std.clone().clamp_min(.01))
        self.net=nn.Sequential(nn.Linear(len(obs_mean)+action_dim,256),nn.ReLU(),
                               nn.Linear(256,256),nn.ReLU(),nn.Linear(256,1))

    def forward(self, obs, action):
        return self.net(torch.cat([(obs-self.obs_mean)/self.obs_std,action],-1)).squeeze(-1)


def make_target(value):
    target=copy.deepcopy(value);target.requires_grad_(False)
    return target


def update_value(value, target, optimizer, actor, obs, actions, rewards, next_obs, gamma=.99, tau=.01):
    with torch.no_grad():
        # Continuing task: success and time-limit truncation both bootstrap.
        expected=rewards+gamma*target(next_obs,actor.get_eval_action(next_obs))
    prediction=value(obs,actions);loss=(prediction-expected).square().mean()
    optimizer.zero_grad();loss.backward();optimizer.step()
    with torch.no_grad():
        for dst,src in zip(target.parameters(),value.parameters()):dst.lerp_(src,tau)
    return loss.detach()
