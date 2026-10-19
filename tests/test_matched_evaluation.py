"""CPU behavioral test: policy RNG consumption must not change later reset batches."""
import json
from types import SimpleNamespace

import torch

from scripts.sac_evidence import evaluate_policy


class Environment:
    num_envs = 2

    def __init__(self):
        self.resets = []
        self.start_observations = []

    def reset(self, seed):
        torch.manual_seed(seed)
        obs = torch.rand(2, 3)
        self.resets.append(seed)
        self.start_observations.append(obs.clone())
        return obs, {}

    def step(self, action):
        # One-step episodes with an RNG-consuming unseeded automatic reset.
        obs = torch.rand(2, 3)
        episode = dict(success_once=torch.zeros(2), success_at_end=torch.zeros(2),
                       episode_len=torch.ones(2), **{'return': torch.zeros(2)})
        return obs, torch.zeros(2), torch.zeros(2), torch.ones(2), {
            '_final_info': torch.ones(2, dtype=torch.bool), 'final_info': {'episode': episode}}


class Policy(torch.nn.Module):
    def __init__(self, draws):
        super().__init__(); self.draws = draws

    def get_eval_action(self, obs):
        torch.rand(self.draws)
        return torch.zeros(2, 1)


def test_matched_batches_and_training_rng_restored(tmp_path):
    args = SimpleNamespace(eval_seed=10000, num_eval_envs=2, num_eval_steps=3,
                           max_episode_steps=1, matched_eval_resets=True)
    envs = []
    for draws in (0, 100):
        output = tmp_path / str(draws); output.mkdir()
        torch.manual_seed(900 + draws); before = torch.get_rng_state().clone()
        env = Environment(); envs.append(env)
        assert evaluate_policy(Policy(draws), env, args, output, 0) == 6
        assert torch.equal(torch.get_rng_state(), before)
        assert env.resets == [10000, 10002, 10004]
        rows = [json.loads(x) for x in (output / 'validation_episodes.jsonl').read_text().splitlines()]
        assert [r['eval_seed'] for r in rows] == [10000, 10000, 10002, 10002, 10004, 10004]
    assert all(torch.equal(a, b) for a, b in zip(envs[0].start_observations, envs[1].start_observations))
