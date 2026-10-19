"""State-only ACT from locked ManiSkill; raw state/actions, masked episode tails.

Original unmodified model/transformer and licenses live in vendor/maniskill/act.
Training adds a padding mask; the decoder exposes only its final layer so that
upstream's [0] indexing selects that layer rather than the first intermediate.
"""
from dataclasses import dataclass
from pathlib import Path
import sys

import h5py
import numpy as np
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'vendor/maniskill/act'))
from act.detr.detr_vae import DETRVAE, build_encoder, reparametrize
from act.detr.transformer import build_transformer


@dataclass
class ACTConfig:
    hidden_dim: int = 128
    dim_feedforward: int = 512
    enc_layers: int = 2
    dec_layers: int = 2
    nheads: int = 4
    dropout: float = 0.1
    num_queries: int = 8
    pre_norm: bool = False
    kl_weight: float = 10.0


class PaddedDETRVAE(DETRVAE):
    def forward(self, obs, actions=None, is_pad=None):
        if actions is None:
            return super().forward(obs)
        batch = obs.shape[0]
        cls = self.cls_embed.weight.unsqueeze(0).expand(batch, -1, -1)
        state = self.encoder_state_proj(obs).unsqueeze(1)
        action = self.encoder_action_proj(actions)
        inputs = torch.cat([cls, state, action], dim=1).transpose(0, 1)
        mask = torch.cat([torch.zeros(batch, 2, dtype=torch.bool, device=obs.device), is_pad], dim=1)
        encoded = self.encoder(inputs, pos=self.pos_table.detach().transpose(0, 1),
                               src_key_padding_mask=mask)[0]
        mu, logvar = self.latent_proj(encoded).chunk(2, dim=-1)
        latent = self.latent_out_proj(reparametrize(mu, logvar))
        state = self.input_proj_robot_state(obs)
        hidden = self.transformer(None, None, self.query_embed.weight, None,
                                  latent, state, self.additional_pos_embed.weight)[0]
        return self.action_head(hidden), (mu, logvar)


class ACTTeacher(nn.Module):
    def __init__(self, state_dim, action_dim, config=None):
        super().__init__()
        self.config = config or ACTConfig()
        self.model = PaddedDETRVAE(None, build_transformer(self.config),
            build_encoder(self.config), state_dim, action_dim, self.config.num_queries)
        self.model.transformer.decoder.return_intermediate = False

    def raw_chunk(self, obs):
        # No posterior actions, latent noise or history at deployment.
        return self.model(obs)[0]

    def chunk(self, obs):
        return self.raw_chunk(obs).clamp(-1, 1)

    def forward(self, obs):
        # Mode A: re-query current observation every real step, execute only a[0].
        return self.chunk(obs)[:, 0]

    def loss(self, obs, actions, is_pad):
        pred, (mu, logvar) = self.model(obs, actions, is_pad)
        valid = (~is_pad).unsqueeze(-1)
        l1 = ((pred-actions).abs()*valid).sum() / (valid.sum()*actions.shape[-1])
        kl = (-0.5*(1+logvar-mu.square()-logvar.exp())).sum(-1).mean()
        return dict(loss=l1+self.config.kl_weight*kl, l1=l1, kl=kl)


class EpisodeChunks:
    """Flattened audited states with explicit episode ends; never cross an episode."""
    def __init__(self, observations, actions, chunk_size, device='cpu'):
        assert len(observations) == len(actions) > 0 and chunk_size > 0
        assert all(len(o) == len(a)+1 and len(a) > 0 for o, a in zip(observations, actions))
        self.obs = torch.as_tensor(np.concatenate([o[:-1] for o in observations]),
                                   dtype=torch.float32, device=device)
        self.actions = torch.as_tensor(np.concatenate(actions), dtype=torch.float32, device=device)
        ends = np.cumsum([len(a) for a in actions])
        self.ends = torch.as_tensor(np.repeat(ends, [len(a) for a in actions]), device=device)
        self.offsets = torch.arange(chunk_size, device=device)

    @classmethod
    def from_h5(cls, path, episode_ids, chunk_size, device='cpu'):
        with h5py.File(path, 'r') as f:
            obs = [f[f'traj_{i}/obs'][:] for i in episode_ids]
            actions = [f[f'traj_{i}/actions'][:] for i in episode_ids]
        return cls(obs, actions, chunk_size, device)

    def __len__(self):
        return len(self.obs)

    def batch(self, indices):
        positions = indices[:, None]+self.offsets[None]
        pad = positions >= self.ends[indices, None]
        positions = torch.minimum(positions, self.ends[indices, None]-1)
        actions = self.actions[positions].masked_fill(pad[..., None], 0)
        return self.obs[indices], actions, pad


class TemporalEnsemble:
    """K bounded predictions per environment; older predictions weight exp(-.01*i).

    Validity is explicit, so zero arm actions are not mistaken for empty slots.
    reset(mask) clears only the completed environments, including partial resets.
    """
    def __init__(self, num_envs, chunk_size, action_dim, device):
        self.values = torch.zeros(num_envs, chunk_size, chunk_size, action_dim, device=device)
        self.valid = torch.zeros(num_envs, chunk_size, dtype=torch.bool, device=device)

    def reset(self, done=None):
        if done is None:
            self.valid.zero_(); self.values.zero_()
        else:
            self.valid[done] = False; self.values[done] = 0

    def update(self, raw_chunk):
        # Rows oldest -> newest; each surviving row's action horizon advances once.
        self.values[:, :-1, :-1] = self.values[:, 1:, 1:].clone()
        self.values[:, :-1, -1] = 0
        self.valid[:, :-1] = self.valid[:, 1:].clone()
        self.values[:, -1] = raw_chunk; self.valid[:, -1] = True
        rank = self.valid.long().cumsum(1)-1
        weights = torch.exp(-.01*rank)*self.valid
        weights = weights/weights.sum(1, keepdim=True)
        return (self.values[:, :, 0]*weights[..., None]).sum(1).clamp(-1, 1)
