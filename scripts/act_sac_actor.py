"""Independent ACT inference-parameter transfer to single-step tanh-Gaussian SAC.

Copied linear action head predicts bounded actions, not pre-tanh means. We map
its clipped output through atanh explicitly; this preserves the teacher mean
within epsilon without treating CVAE latent samples as policy log probability.
"""
import copy
import hashlib
import math
import torch
from torch import nn
from scripts.act_teacher import ACTConfig, build_transformer


class ACTSACActor(nn.Module):
    copied_modules = ('transformer', 'input_proj_robot_state', 'latent_out_proj',
                      'additional_pos_embed', 'query_embed', 'action_head')

    def __init__(self, config=None, state_dim=42, action_dim=4, log_std=-2.0, action_epsilon=1e-6):
        super().__init__()
        self.config = copy.deepcopy(config or ACTConfig())
        self.action_epsilon = action_epsilon
        self.latent_dim = 32
        hidden = self.config.hidden_dim
        self.transformer = build_transformer(self.config)
        self.transformer.decoder.return_intermediate = False
        self.input_proj_robot_state = nn.Linear(state_dim, hidden)
        self.latent_out_proj = nn.Linear(self.latent_dim, hidden)
        self.additional_pos_embed = nn.Embedding(2, hidden)
        self.query_embed = nn.Embedding(self.config.num_queries, hidden)
        self.action_head = nn.Linear(hidden, action_dim)
        self.fc_logstd = nn.Linear(hidden, action_dim)
        with torch.no_grad():
            self.fc_logstd.weight.zero_()
            self.fc_logstd.bias.fill_(math.atanh(2*(log_std+5)/7-1))
        self.register_buffer('action_scale', torch.ones(action_dim))
        self.register_buffer('action_bias', torch.zeros(action_dim))
        self.requires_grad_(True)
        self.train()

    @classmethod
    def from_teacher(cls, teacher, log_std=-2.0, action_epsilon=1e-6):
        actor = cls(teacher.config, teacher.model.input_proj_robot_state.in_features,
                    teacher.model.action_head.out_features, log_std, action_epsilon)
        for name in cls.copied_modules:
            getattr(actor, name).load_state_dict(getattr(teacher.model, name).state_dict(), strict=True)
        return actor

    def train(self, mode=True):
        super().train(mode)
        # The conditional Gaussian must not hide a second dropout distribution.
        # eval affects dropout only here, not parameter gradients/optimization.
        self.transformer.eval()
        return self

    def features(self, obs):
        latent = self.latent_out_proj(obs.new_zeros(len(obs), self.latent_dim))
        state = self.input_proj_robot_state(obs)
        hidden = self.transformer(None, None, self.query_embed.weight, None,
            latent, state, self.additional_pos_embed.weight)[0]
        return hidden[:, 0]  # all K queries interact before selecting the current one

    def forward(self, obs):
        hidden = self.features(obs)
        bounded = self.action_head(hidden).clamp(-1+self.action_epsilon, 1-self.action_epsilon)
        mean = torch.atanh(bounded)
        log_std = -5 + 3.5*(torch.tanh(self.fc_logstd(hidden))+1)
        return mean, log_std

    def get_eval_action(self, obs):
        mean, _ = self(obs)
        return mean.tanh()*self.action_scale+self.action_bias

    def get_action(self, obs):
        mean, log_std = self(obs)
        normal = torch.distributions.Normal(mean, log_std.exp())
        pre_tanh = normal.rsample(); bounded = pre_tanh.tanh()
        action = bounded*self.action_scale+self.action_bias
        # Same unit-action tanh correction and epsilon as the existing ManiSkill SAC.
        log_prob = (normal.log_prob(pre_tanh)-torch.log(self.action_scale*(1-bounded.square())+1e-6)).sum(1, keepdim=True)
        return action, log_prob, mean.tanh()*self.action_scale+self.action_bias

    def copy_audit(self, teacher):
        source = teacher.model.state_dict(); target = self.state_dict(); records=[]
        for name, value in target.items():
            if name.split('.')[0] not in self.copied_modules: continue
            origin = source[name]
            assert value.shape==origin.shape and value.dtype==origin.dtype
            assert torch.equal(value.cpu(), origin.cpu()) and value.data_ptr()!=origin.data_ptr()
            records.append(dict(source='model.'+name,target=name,shape=list(value.shape),dtype=str(value.dtype),
                sha256=hashlib.sha256(value.detach().cpu().contiguous().numpy().tobytes()).hexdigest(),shared_storage=False))
        return records
