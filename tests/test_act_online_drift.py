import hashlib
import json

import h5py
import numpy as np
import pytest
import torch

from scripts.diagnose_act_online_drift import gaussian_kl, train_probe


def test_gaussian_kl_matches_distribution_and_identity():
    torch.manual_seed(1)
    a = dict(pre_tanh_mean=torch.randn(7, 4), log_std=torch.randn(7, 4))
    b = dict(pre_tanh_mean=torch.randn(7, 4), log_std=torch.randn(7, 4))
    reference = torch.distributions.kl_divergence(
        torch.distributions.Normal(a['pre_tanh_mean'], a['log_std'].exp()),
        torch.distributions.Normal(b['pre_tanh_mean'], b['log_std'].exp())).sum(-1)
    assert torch.allclose(gaussian_kl(a, b), reference, atol=1e-5, rtol=1e-5)
    assert torch.equal(gaussian_kl(a, a), torch.zeros(7))


def test_probe_uses_only_training_pre_action_states(tmp_path):
    dataset = tmp_path/'demos.h5'
    with h5py.File(dataset, 'w') as f:
        for episode in [3, 8, 19]:
            group = f.create_group(f'traj_{episode}')
            obs = np.full((4, 42), episode, dtype=np.float32)
            obs[-1] = 999  # terminal observation has no aligned expert action
            group['obs'] = obs
            group['actions'] = np.zeros((3, 4), dtype=np.float32)
    audit = dict(dataset=str(dataset), dataset_sha256=hashlib.sha256(dataset.read_bytes()).hexdigest(),
                 train_episode_ids=[8, 3], validation_episode_ids=[19], alignment_validated=True)
    path = tmp_path/'audit.json'; path.write_text(json.dumps(audit))
    obs, ids, times, _ = train_probe(path, count=5)
    assert ids.tolist() == [8, 8, 8, 3, 3]
    assert times.tolist() == [0, 1, 2, 0, 1]
    assert obs[:, 0].tolist() == [8, 8, 8, 3, 3]
    audit['validation_episode_ids'] = [3]
    path.write_text(json.dumps(audit))
    with pytest.raises(AssertionError):
        train_probe(path, count=5)
