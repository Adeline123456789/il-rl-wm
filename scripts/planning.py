"""Batched H=2 policy candidates and a common pure-task return score."""
import torch


@torch.no_grad()
def generate_candidates(obs, actor, world_model, il=None, deterministic=False):
    n_sac=8 if il is not None else 9
    n=9;batch,state_dim=obs.shape
    state=obs[:,None,:].expand(batch,n,state_dim)
    states=[state];actions=[];rewards=[]
    for _ in range(2):
        sac_state=state[:,(1 if il is not None else 0):].reshape(batch*n_sac,state_dim)
        sac_action=actor.get_eval_action(sac_state) if deterministic else actor.get_action(sac_state)[0]
        action=sac_action.reshape(batch,n_sac,-1)
        if il is not None:action=torch.cat([il(state[:,0])[:,None,:],action],dim=1)
        flat_state,logit=world_model(state.reshape(batch*n,state_dim),action.reshape(batch*n,-1))
        state=flat_state.reshape(batch,n,state_dim)
        actions.append(action);rewards.append(logit.sigmoid().reshape(batch,n));states.append(state)
    return dict(actions=torch.stack(actions,2),states=torch.stack(states,2),
                rewards=torch.stack(rewards,2),sources=['il']+['sac']*8 if il is not None else ['sac']*9)


def discounted_score(rewards, terminal_value, gamma):
    horizon=rewards.shape[-1]
    discounts=gamma**torch.arange(horizon,device=rewards.device)
    short_return=(rewards*discounts).sum(-1)
    terminal_component=gamma**horizon*terminal_value
    return short_return+terminal_component,short_return,terminal_component


@torch.no_grad()
def score_candidates(candidates, actor, task_value, gamma=.99):
    terminal=candidates['states'][:,:,-1]
    flat=terminal.flatten(0,1)
    value=task_value(flat,actor.get_eval_action(flat)).reshape(terminal.shape[:2])
    score,short_return,terminal_component=discounted_score(candidates['rewards'],value,gamma)
    selected=score.argmax(1)
    first_actions=candidates['actions'][torch.arange(len(selected),device=selected.device),selected,0]
    return dict(action=first_actions,selected=selected,score=score,short_return=short_return,
                terminal_value=value,terminal_component=terminal_component)
