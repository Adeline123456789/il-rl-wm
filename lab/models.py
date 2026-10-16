"""Small SAC networks; SAC update conventions follow CleanRL (see vendor).

World-model/teacher integration is project code, not a claimed novel SAC algorithm.
"""
import copy
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


def mlp(input_dim, output_dim):
    return nn.Sequential(nn.Linear(input_dim, 256), nn.ReLU(),
                         nn.Linear(256, 256), nn.ReLU(), nn.Linear(256, output_dim))


class Actor(nn.Module):
    def __init__(self, obs_dim, action_dim):
        super().__init__()
        self.features = nn.Sequential(nn.Linear(obs_dim, 256), nn.ReLU(), nn.Linear(256, 256), nn.ReLU())
        self.mean = nn.Linear(256, action_dim)
        self.log_std = nn.Linear(256, action_dim)
        nn.init.zeros_(self.log_std.weight)
        nn.init.constant_(self.log_std.bias, -1.0)

    def forward(self, obs, deterministic=False):
        h = self.features(obs)
        mean = self.mean(h)
        if deterministic:
            return mean.tanh()
        log_std = self.log_std(h).clamp(-5, 2)
        normal = torch.distributions.Normal(mean, log_std.exp())
        u = normal.rsample()
        a = u.tanh()
        log_prob = (normal.log_prob(u) - torch.log(1 - a.square() + 1e-6)).sum(-1, keepdim=True)
        return a, log_prob


class Q(nn.Module):
    def __init__(self, obs_dim, action_dim):
        super().__init__()
        self.net = mlp(obs_dim + action_dim, 1)

    def forward(self, obs, action):
        return self.net(torch.cat([obs, action], -1))


class WorldModel(nn.Module):
    def __init__(self, obs_dim, action_dim):
        super().__init__()
        self.obs_dim = obs_dim
        self.net = mlp(obs_dim + action_dim, obs_dim + 2)

    def forward(self, obs, action):
        pred = self.net(torch.cat([obs, action], -1))
        return obs + pred[..., :self.obs_dim], pred[..., -2:-1], pred[..., -1:]


class Replay:
    def __init__(self, obs_dim, action_dim, capacity=200000):
        self.capacity = capacity
        self.size = self.pos = 0
        self.arrays = [np.empty((capacity, d), np.float32) for d in [obs_dim, action_dim, 1, obs_dim, 1]]

    def add(self, obs, action, reward, next_obs, terminated):
        for target, value in zip(self.arrays, [obs, action, reward, next_obs, terminated]):
            target[self.pos] = value
        self.pos = (self.pos + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, n, device, generator):
        idx = generator.integers(self.size, size=n)
        return [torch.as_tensor(v[idx], device=device) for v in self.arrays]


class SAC:
    def __init__(self, obs_dim, action_dim, device, teacher=None, use_world_model=False):
        self.device = device
        self.actor = Actor(obs_dim, action_dim).to(device)
        if teacher is not None:
            self.actor.load_state_dict(teacher.state_dict())
        self.teacher = copy.deepcopy(teacher).eval().requires_grad_(False) if teacher is not None else None
        self.q = nn.ModuleList([Q(obs_dim, action_dim), Q(obs_dim, action_dim)]).to(device)
        self.target_q = copy.deepcopy(self.q).requires_grad_(False)
        self.qr = nn.ModuleList([Q(obs_dim, action_dim), Q(obs_dim, action_dim)]).to(device)
        self.target_qr = copy.deepcopy(self.qr).requires_grad_(False)
        self.wm = WorldModel(obs_dim, action_dim).to(device) if use_world_model else None
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=3e-4)
        self.q_opt = torch.optim.Adam(self.q.parameters(), lr=3e-4)
        self.qr_opt = torch.optim.Adam(self.qr.parameters(), lr=3e-4)
        self.wm_opt = torch.optim.Adam(self.wm.parameters(), lr=3e-4) if self.wm else None
        self.log_alpha = torch.zeros(1, requires_grad=True, device=device)
        self.alpha_opt = torch.optim.Adam([self.log_alpha], lr=3e-4)
        self.target_entropy = -action_dim
        self.gamma, self.tau = 0.99, 0.005
        self.teacher_calls = 0

    def update(self, batch):
        obs, a, reward, next_obs, terminated = batch
        alpha = self.log_alpha.exp().detach()
        with torch.no_grad():
            na, nlp = self.actor(next_obs)
            y = reward + self.gamma * (1 - terminated) * (torch.minimum(*[q(next_obs, na) for q in self.target_q]) - alpha * nlp)
            yr = reward + self.gamma * (1 - terminated) * torch.minimum(*[q(next_obs, na) for q in self.target_qr])
        q_loss = sum(F.mse_loss(q(obs, a), y) for q in self.q)
        self.q_opt.zero_grad(); q_loss.backward(); self.q_opt.step()
        qr_loss = sum(F.mse_loss(q(obs, a), yr) for q in self.qr)
        self.qr_opt.zero_grad(); qr_loss.backward(); self.qr_opt.step()
        self.q.requires_grad_(False)
        action, lp = self.actor(obs)
        actor_loss = (alpha * lp - torch.minimum(*[q(obs, action) for q in self.q])).mean()
        self.actor_opt.zero_grad(); actor_loss.backward(); self.actor_opt.step()
        self.q.requires_grad_(True)
        alpha_loss = -(self.log_alpha.exp() * (lp.detach() + self.target_entropy)).mean()
        self.alpha_opt.zero_grad(); alpha_loss.backward(); self.alpha_opt.step()
        wm_loss = torch.zeros((), device=obs.device)
        if self.wm is not None:
            ns, rr, end = self.wm(obs, a)
            wm_loss = F.mse_loss(ns, next_obs) + F.mse_loss(rr, reward) + F.binary_cross_entropy_with_logits(end, terminated)
            self.wm_opt.zero_grad(); wm_loss.backward(); self.wm_opt.step()
        with torch.no_grad():
            for target, source in [(self.target_q, self.q), (self.target_qr, self.qr)]:
                for tp, p in zip(target.parameters(), source.parameters()):
                    tp.lerp_(p, self.tau)
        metrics = {"q_loss": q_loss.item(), "qr_loss": qr_loss.item(), "actor_loss": actor_loss.item(),
                   "alpha": self.log_alpha.exp().item(), "wm_loss": wm_loss.item()}
        if not all(np.isfinite(x) for x in metrics.values()):
            raise FloatingPointError(f"Non-finite update: {metrics}")
        return metrics

    @torch.no_grad()
    def action(self, obs, mode="sac", deterministic=False):
        x = torch.as_tensor(obs, dtype=torch.float32, device=self.device).reshape(1, -1)
        if mode == "il":
            if self.teacher is None: raise ValueError("没有教师")
            self.teacher_calls += 1
            return self.teacher(x, deterministic=True)[0].cpu().numpy()
        if mode == "sac":
            a = self.actor(x, deterministic=True) if deterministic else self.actor(x)[0]
            return a[0].cpu().numpy()
        if self.wm is None: raise ValueError("规划模式需要世界模型")
        use_teacher = mode == "dual"
        candidates = 9
        states = x.expand(candidates, -1).clone()
        score = torch.zeros((candidates, 1), device=self.device)
        alive = torch.ones_like(score)
        first = None
        for t in range(2):
            aa = self.actor(states)[0]
            if use_teacher:
                if self.teacher is None: raise ValueError("双策略模式需要冻结教师")
                self.teacher_calls += 1
                aa[0] = self.teacher(states[:1], deterministic=True)[0]
            if t == 0: first = aa.clone()
            states, rr, end = self.wm(states, aa)
            score += self.gamma ** t * alive * rr
            alive *= 1 - end.sigmoid()
        tail_a, _ = self.actor(states)
        score += self.gamma ** 2 * alive * torch.minimum(*[q(states, tail_a) for q in self.qr])
        return first[score.argmax().item()].cpu().numpy()

    def checkpoint(self):
        return {"actor": self.actor.state_dict(), "q": self.q.state_dict(), "target_q": self.target_q.state_dict(),
                "qr": self.qr.state_dict(), "target_qr": self.target_qr.state_dict(),
                "world_model": self.wm.state_dict() if self.wm else None,
                "teacher": self.teacher.state_dict() if self.teacher else None,
                "actor_optimizer": self.actor_opt.state_dict(), "q_optimizer": self.q_opt.state_dict(),
                "qr_optimizer": self.qr_opt.state_dict(), "wm_optimizer": self.wm_opt.state_dict() if self.wm_opt else None,
                "log_alpha": self.log_alpha.detach(), "alpha_optimizer": self.alpha_opt.state_dict(),
                "torch_rng": torch.get_rng_state(), "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}
