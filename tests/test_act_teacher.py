import numpy as np
import torch

from scripts.act_teacher import ACTConfig, ACTTeacher, EpisodeChunks
from scripts.policy_bridge import freeze_il, fingerprint


def teacher():
    return ACTTeacher(6, 4, ACTConfig(hidden_dim=32, dim_feedforward=64,
        enc_layers=1, dec_layers=1, dropout=0, num_queries=4))


def test_chunks_never_cross_episode_and_keep_first_action():
    ds = EpisodeChunks([np.zeros((4, 6)), np.ones((3, 6))],
        [np.arange(12).reshape(3, 4), np.full((2, 4), 99)], 4)
    obs, actions, pad = ds.batch(torch.arange(5))
    assert torch.equal(actions[:, 0], ds.actions)
    assert pad.sum(1).tolist() == [1, 2, 3, 2, 3]
    assert not (actions[:3] == 99).any()
    assert (actions[pad] == 0).all()
    assert obs.shape == (5, 6)


def test_pad_does_not_change_posterior_or_loss():
    torch.manual_seed(7)
    model = teacher().eval()
    obs, actions = torch.randn(3, 6), torch.randn(3, 4, 4)
    pad = torch.tensor([[False, True, True, True], [False, False, True, True], [False]*4])
    changed = actions.clone(); changed[pad] = 12345
    torch.manual_seed(11); first = model.loss(obs, actions, pad)
    torch.manual_seed(11); second = model.loss(obs, changed, pad)
    for key in first:
        torch.testing.assert_close(first[key], second[key], rtol=0, atol=0)
    first['loss'].backward()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)


def test_prior_is_deterministic_bounded_and_teacher_independent():
    torch.manual_seed(5); model = teacher().eval()
    torch.manual_seed(5); same = teacher().eval()
    assert fingerprint(model) == fingerprint(same)
    obs = torch.randn(3, 6)
    before = torch.get_rng_state().clone()
    chunk = model.chunk(obs)
    assert torch.equal(before, torch.get_rng_state())
    assert chunk.shape == (3, 4, 4) and chunk.abs().max() <= 1
    assert torch.equal(model(obs), chunk[:, 0])
    frozen = freeze_il(model); expected = fingerprint(frozen)
    with torch.no_grad(): next(model.parameters()).add_(1)
    assert fingerprint(frozen) == expected


def test_unpadded_adaptation_matches_upstream_exactly():
    from act.detr.detr_vae import DETRVAE
    model = teacher().model.eval()
    obs, actions = torch.randn(2, 6), torch.randn(2, 4, 4)
    torch.manual_seed(13); expected = DETRVAE.forward(model, obs, actions)
    torch.manual_seed(13); actual = model(obs, actions, torch.zeros(2, 4, dtype=torch.bool))
    assert torch.equal(expected[0], actual[0])
    assert all(torch.equal(x, y) for x, y in zip(expected[1], actual[1]))


def test_all_decoder_layers_receive_learning_signal():
    model = ACTTeacher(6, 4, ACTConfig(hidden_dim=32, dim_feedforward=64,
        enc_layers=2, dec_layers=2, dropout=0, num_queries=4))
    loss = model.loss(torch.randn(3, 6), torch.randn(3, 4, 4),
                      torch.zeros(3, 4, dtype=torch.bool))['loss']
    loss.backward()
    for layer in model.model.transformer.decoder.layers:
        assert layer.linear1.weight.grad is not None
        assert layer.linear1.weight.grad.norm() > 0


def test_temporal_ensemble_matches_explicit_history_and_partial_reset():
    from scripts.act_teacher import TemporalEnsemble
    memory = TemporalEnsemble(2, 4, 1, 'cpu')
    history = [[], []]
    for step in range(9):
        if step == 3:
            memory.reset(torch.tensor([True, False])); history[0] = []
        chunk = torch.arange(8).reshape(2, 4, 1).float()/100 + step/20
        actual = memory.update(chunk)
        for env in range(2):
            history[env].append(chunk[env].clone())
            retained = history[env][-4:]
            target = torch.stack([c[len(retained)-i-1] for i,c in enumerate(retained)])
            weights = torch.exp(-.01*torch.arange(len(retained)))
            expected = (target*weights[:, None]).sum(0)/weights.sum()
            torch.testing.assert_close(actual[env], expected)
    memory.reset(); zeros = torch.zeros(2,4,1)
    assert torch.equal(memory.update(zeros), torch.zeros(2,1))
