"""Expert replay is independent of actor initialization; no simulator is created."""
from dataclasses import fields
import json
from pathlib import Path

import pytest

from scripts.train_sac import Args, validate_args


def registered_args(condition):
    plan = json.loads(Path('configs/act_next_initialization_controls.json').read_text())
    config = next(x['config'] for x in plan['conditions'] if x['id'] == condition)
    names = {field.name for field in fields(Args)}
    return Args(**{key: value for key, value in config.items() if key in names})


@pytest.mark.parametrize('condition', ['R0', 'R1', 'R2', 'R3'])
def test_registered_initialization_controls_accept_expert_replay(condition):
    args = registered_args(condition)
    validate_args(args)
    if condition == 'R0':
        assert args.bc_checkpoint is None and args.actor_init_checkpoint is None
        assert args.expert_dataset and args.expert_audit


def test_random_actor_expert_replay_still_requires_audit_and_equal_split():
    args = registered_args('R0')
    args.expert_audit = None
    with pytest.raises(AssertionError):
        validate_args(args)
    args = registered_args('R0')
    args.batch_size = 1023
    with pytest.raises(AssertionError, match='exactly 50%'):
        validate_args(args)


def test_random_expert_control_does_not_bypass_planning_requirements():
    args = registered_args('R0')
    args.wm_checkpoint = 'unloaded-world-model.pt'
    args.task_value_checkpoint = 'unloaded-value.pt'
    with pytest.raises(AssertionError):
        validate_args(args)
