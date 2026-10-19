import torch
from scripts.act_teacher import ACTTeacher, ACTConfig
from scripts.act_sac_actor import ACTSACActor
from scripts.policy_bridge import fingerprint


def test_parameter_transfer_gaussian_alignment_and_independence():
    torch.set_num_threads(2);torch.manual_seed(7)
    teacher=ACTTeacher(42,4,ACTConfig(num_queries=8)).eval().requires_grad_(False)
    before=fingerprint(teacher)
    actor=ACTSACActor.from_teacher(teacher)
    records=actor.copy_audit(teacher);assert len(records)>50
    obs=torch.randn(16,42)
    assert torch.allclose(actor.get_eval_action(obs),teacher(obs),atol=2e-6,rtol=1e-5)
    assert torch.equal(actor.get_eval_action(obs),actor.get_eval_action(obs))
    assert not any(n.startswith('encoder.') for n,_ in actor.named_parameters())
    sample,lp,mean=actor.get_action(obs)
    assert torch.isfinite(lp).all() and sample.abs().max()<=1
    assert torch.equal(mean,actor.get_eval_action(obs))
    optimizer=torch.optim.Adam(actor.parameters(),lr=1e-4)
    loss=(.2*lp-sample[:, :1]).mean()
    optimizer.zero_grad();loss.backward()
    assert actor.action_head.weight.grad.isfinite().all()
    assert actor.transformer.decoder.layers[1].linear1.weight.grad.abs().sum()>0
    assert actor.fc_logstd.weight.grad.abs().sum()>0
    optimizer.step()
    assert fingerprint(teacher)==before and all(p.grad is None for p in teacher.parameters())


def test_saturated_action_has_finite_pre_tanh_and_probability():
    teacher=ACTTeacher(42,4).eval().requires_grad_(False)
    with torch.no_grad():teacher.model.action_head.weight.zero_();teacher.model.action_head.bias.fill_(2)
    actor=ACTSACActor.from_teacher(teacher);obs=torch.zeros(8,42)
    assert torch.isfinite(actor(obs)[0]).all()
    assert torch.allclose(actor.get_eval_action(obs),teacher(obs),atol=2e-6,rtol=0)
    sample,lp,_=actor.get_action(obs);assert torch.isfinite(lp).all()
    lp.mean().backward();assert actor.fc_logstd.bias.grad.isfinite().all()


def test_reload_without_constructing_teacher(monkeypatch):
    teacher=ACTTeacher(42,4,ACTConfig(num_queries=8)).eval()
    actor=ACTSACActor.from_teacher(teacher).eval()
    def forbidden(*args,**kwargs):raise AssertionError('Teacher construction forbidden')
    monkeypatch.setattr(ACTTeacher,'__init__',forbidden)
    restored=ACTSACActor(actor.config).eval();restored.load_state_dict(actor.state_dict(),strict=True)
    obs=torch.randn(4,42)
    assert torch.equal(actor.get_eval_action(obs),restored.get_eval_action(obs))
