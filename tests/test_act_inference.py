"""CPU-only inference/gate contracts; these do not simulate robot success."""
from dataclasses import asdict
import json

import pytest
import torch

from scripts.act_inference import action_callable, load_model
from scripts.act_sac_actor import ACTSACActor
from scripts.act_teacher import ACTConfig, ACTTeacher
from scripts.distill_act_sac import make_student
from scripts.evaluate_act_checkpoint import validate_protocol
from scripts.train_act_teacher import closed_loop


def test_final_gate_and_registered_protocol(tmp_path):
    gate = tmp_path / 'gate.json'
    gate.write_text(json.dumps(dict(released=False)))
    validate_protocol('teacher', 'first', 10000, 40, 8, gate)
    with pytest.raises(AssertionError, match='gate closed'):
        validate_protocol('act', 'mean', 20000, 100, 10, gate)
    gate.write_text(json.dumps(dict(released=True)))
    validate_protocol('act', 'stochastic', 20000, 100, 10, gate)
    for values in [('teacher', 'mean', 10000, 40, 8),
                   ('mlp', 'ensemble', 10000, 40, 8),
                   ('act', 'mean', 20000, 40, 8),
                   ('act', 'mean', 10000, 41, 8)]:
        with pytest.raises(AssertionError):
            validate_protocol(*values, gate)


def test_closed_loop_rejects_final_before_creating_environment(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / 'configs').mkdir()
    (tmp_path / 'configs/act_final_test_gate.json').write_text('{"released": false}')
    with pytest.raises(AssertionError, match='gate closed'):
        closed_loop(None, tmp_path, 0, 100, None,
                    eval_seed=20000, num_envs=10, purpose='final_test')


@pytest.mark.parametrize('kind', ['mlp', 'act'])
def test_student_reload_preserves_learned_variance_without_teacher(kind, monkeypatch):
    torch.set_num_threads(2)
    torch.manual_seed(23)
    if kind == 'mlp':
        original = make_student().eval()
        saved = {}
    else:
        config = ACTConfig(num_queries=8)
        original = ACTSACActor(config).eval()
        saved = dict(model_config=asdict(config), state_dim=42, action_dim=4,
                     log_std_init=-2., action_epsilon=1e-6)
    # A trained variance must survive loading; do not silently reinitialize it.
    with torch.no_grad():
        original.fc_logstd.weight.normal_(0, .01)
        original.fc_logstd.bias.fill_(.15)
    saved['actor'] = original.state_dict()
    def forbidden(*args, **kwargs):
        raise AssertionError('Teacher construction forbidden')
    monkeypatch.setattr(ACTTeacher, '__init__', forbidden)
    restored = load_model(saved, kind)
    assert not restored.training and not any(p.requires_grad for p in restored.parameters())
    assert all(torch.equal(value, restored.state_dict()[key]) for key, value in saved['actor'].items())
    assert not {p.data_ptr() for p in original.parameters()} & {p.data_ptr() for p in restored.parameters()}
    obs = torch.randn(8, 42)
    with torch.no_grad():
        assert torch.equal(action_callable(restored, kind, 'mean')(obs), original.get_eval_action(obs))
        torch.manual_seed(42)
        expected = original.get_action(obs)[0]
        torch.manual_seed(42)
        assert torch.equal(action_callable(restored, kind, 'stochastic')(obs), expected)
