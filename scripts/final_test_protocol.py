"""Check an explicit frozen final-test registry before simulator construction."""
import json
from pathlib import Path


def validate_final_test(seed, episodes, num_envs, gate, protocol=None):
    assert (episodes, num_envs) == (100, 10)
    if protocol is None:
        assert seed == 20000
    else:
        registered = json.loads(Path(protocol).read_text())
        assert registered['status'] == 'registered_before_training'
        assert (seed, episodes, num_envs) == (
            registered['seed'], registered['episodes'], registered['num_envs'])
        assert seed not in (10000, 20000)
        gate = registered['gate']
    assert json.loads(Path(gate).read_text())['released'], 'Final-test gate closed'
