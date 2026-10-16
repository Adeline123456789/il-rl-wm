"""Small raw-state residual model; policy interfaces stay in environment units."""
import torch
from torch import nn
from torch.nn import functional as F


class WorldModel(nn.Module):
    def __init__(self, obs_mean, obs_std, action_dim):
        super().__init__()
        self.register_buffer('obs_mean', obs_mean.clone())
        self.register_buffer('obs_std', obs_std.clone().clamp_min(.01))
        self.net = nn.Sequential(nn.Linear(len(obs_mean)+action_dim,256), nn.ReLU(),
                                 nn.Linear(256,256), nn.ReLU(), nn.Linear(256,len(obs_mean)+1))

    def forward(self, obs, action):
        prediction=self.net(torch.cat([(obs-self.obs_mean)/self.obs_std,action],dim=-1))
        return obs+prediction[...,:-1]*self.obs_std, prediction[...,-1]

    def rollout(self, obs, actions):
        states=[];logits=[]
        for action in actions.unbind(1):
            obs, logit=self(obs,action)
            states.append(obs);logits.append(logit)
        return torch.stack(states,1),torch.stack(logits,1)

    def loss(self, states, actions, rewards):
        predicted,logits=self.rollout(states[:,0],actions)
        state_loss=((predicted-states[:,1:])/self.obs_std).square().mean()
        reward_loss=F.binary_cross_entropy_with_logits(logits,rewards)
        return state_loss+reward_loss,dict(state_loss=state_loss.detach(),reward_loss=reward_loss.detach())
