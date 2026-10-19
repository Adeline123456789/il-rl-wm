import torch
from scripts.act_retention import MeanRetention


class Actor(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.mean = torch.nn.Linear(3, 2)
        self.log_std = torch.nn.Linear(3, 2)

    def get_eval_action(self, obs):
        return self.mean(obs).tanh()


def test_retention_only_mean_gradient_and_separate_rng():
    actor = Actor()
    obs = torch.randn(9, 3, requires_grad=True)
    target = torch.ones(9, 2, requires_grad=True)
    retention = MeanRetention(obs, target, 10, 65536, 4, 0)
    rng = torch.get_rng_state().clone()
    loss, weight = retention.loss(actor, 4096)
    (weight * loss).backward()
    assert weight == 9.375 and retention.samples == 4
    assert torch.equal(rng, torch.get_rng_state())
    assert actor.mean.weight.grad.abs().sum() > 0
    assert actor.log_std.weight.grad is None and actor.log_std.bias.grad is None
    assert obs.grad is None and target.grad is None


def test_retention_stops_without_actor_or_rng_access():
    retention = MeanRetention(torch.zeros(9, 3), torch.zeros(9, 2), 10, 65536, 4, 0)
    class NoActor:
        def get_eval_action(self, _):
            raise AssertionError('must not evaluate after cutoff')
    before = retention.generator.get_state().clone()
    for step in [65536, 262144]:
        loss, weight = retention.loss(NoActor(), step)
        assert loss == 0 and weight == 0
    assert torch.equal(before, retention.generator.get_state())
    assert retention.samples == retention.updates == 0
