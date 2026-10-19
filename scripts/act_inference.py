"""Explicit loaders for the three registered state-policy inference formats."""
import torch


def load_model(state, kind, device='cpu'):
    if kind == 'teacher':
        from scripts.act_teacher import ACTConfig, ACTTeacher
        model = ACTTeacher(state['state_dim'], state['action_dim'], ACTConfig(**state['model_config']))
        model.load_state_dict(state['model'], strict=True)
    elif kind == 'act':
        from scripts.act_sac_actor import ACTSACActor
        from scripts.act_teacher import ACTConfig
        model = ACTSACActor(ACTConfig(**state['model_config']), state['state_dim'], state['action_dim'],
                            state['log_std_init'], state['action_epsilon'])
        model.load_state_dict(state['actor'], strict=True)
    else:
        assert kind == 'mlp'
        from scripts.distill_act_sac import make_student
        model = make_student()
        model.load_state_dict(state['actor'], strict=True)
    return model.to(device).eval().requires_grad_(False)


def action_callable(model, kind, mode):
    if kind == 'teacher':
        assert mode in ('first', 'ensemble')
        return model
    assert mode in ('mean', 'stochastic')
    if mode == 'mean':
        return model.get_eval_action
    def sample(obs):
        action, log_prob, _ = model.get_action(obs)
        assert torch.isfinite(log_prob).all()
        return action
    return sample
