"""Reconstruct the fixed architectures from saved weights without a simulator."""
from types import SimpleNamespace
import gymnasium as gym
import numpy as np
import torch
from scripts.world_model import WorldModel
from scripts.task_value import TaskValue


def sac_actor(snapshot, device):
    from scripts.train_sac import Actor as SACActor
    weights=snapshot['actor'];dim=weights['backbone.0.weight'].shape[1]
    scale=weights['action_scale'].cpu().numpy();bias=weights['action_bias'].cpu().numpy()
    env=SimpleNamespace(single_observation_space=gym.spaces.Box(-np.inf,np.inf,shape=(dim,),dtype=np.float32),
                        single_action_space=gym.spaces.Box(bias-scale,bias+scale,dtype=np.float32))
    actor=SACActor(env).to(device);actor.load_state_dict(weights)
    return actor


def bc_actor(path, obs_dim, action_dim, device):
    from scripts.train_bc import Actor as BCActor
    actor=BCActor(obs_dim,action_dim).to(device)
    actor.load_state_dict(torch.load(path,map_location=device)['actor'])
    actor.eval();actor.requires_grad_(False)
    return actor


def world_model(snapshot, device):
    weights=snapshot['model'];dim=len(weights['obs_mean']);action_dim=weights['net.0.weight'].shape[1]-dim
    model=WorldModel(weights['obs_mean'],weights['obs_std'],action_dim).to(device)
    model.load_state_dict(weights)
    return model


def task_value(snapshot, device):
    weights=snapshot['value'];dim=len(weights['obs_mean']);action_dim=weights['net.0.weight'].shape[1]-dim
    model=TaskValue(weights['obs_mean'],weights['obs_std'],action_dim).to(device)
    model.load_state_dict(weights)
    return model
