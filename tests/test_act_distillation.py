import torch
from scripts.distill_act_sac import make_student
from scripts.policy_bridge import fingerprint


def test_distillation_mean_and_variance_contract():
    torch.set_num_threads(2);torch.manual_seed(0)
    student=make_student(-2.0);variance=fingerprint(student.fc_logstd)
    obs=torch.randn(16,42);target=torch.tanh(torch.randn(16,4))
    before=student.get_eval_action(obs).detach().clone()
    optimizer=torch.optim.Adam([p for p in student.parameters() if p.requires_grad],lr=1e-3)
    loss=(student.get_eval_action(obs)-target).square().mean()
    optimizer.zero_grad();loss.backward();optimizer.step()
    assert not torch.equal(before,student.get_eval_action(obs))
    assert fingerprint(student.fc_logstd)==variance
    sampled,lp,mean=student.get_action(obs)
    assert torch.equal(mean,student.get_eval_action(obs))
    assert torch.isfinite(lp).all() and sampled.abs().max()<=1
    assert torch.allclose(student(obs)[1],torch.full((16,4),-2.0))
    (-lp.mean()).backward()
    assert student.fc_mean.weight.grad.isfinite().all()
