"""F1: fixed candidates, repeated matched reset/prefix, two distinct continuations."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import gymnasium as gym
import numpy as np
import torch
import mani_skill.envs
from scripts.load_components import sac_actor,world_model,task_value
from scripts.train_bc import Actor as BCActor
from scripts.planning import generate_candidates,score_candidates
from scripts.interaction_counter import InteractionCounter,CountedEnv


def ranking_metrics(actual, predicted, selected, tolerance=1.):
    """One row per state, one column per frozen candidate; ties are not ranked."""
    rows=[]
    for values,scores,choice in zip(actual,predicted,selected):
        correct=eligible=ties=0
        for i in range(len(values)):
            for j in range(i+1,len(values)):
                difference=values[i]-values[j]
                if abs(difference)<=tolerance:ties+=1;continue
                eligible+=1;correct+=int(np.sign(difference)==np.sign(scores[i]-scores[j]))
        rows.append(dict(candidate_best_return=float(values.max()),selected_return=float(values[choice]),
            regret=float(values.max()-values[choice]),rankable_pairs=eligible,near_tie_pairs=ties,
            pairwise_accuracy=correct/eligible if eligible else None,
            selected_source='il' if choice==0 else 'sac',
            wrong_takeover=bool(choice==0 and values[1:].max()>values[0]+tolerance),
            wrong_rejection=bool(choice!=0 and values[0]>values[1:].max()+tolerance)))
    return rows


def main():
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--states',type=int,default=8);p.add_argument('--prefix-steps',type=int,default=40);p.add_argument('--continuation-steps',type=int,default=400);p.add_argument('--repeats',type=int,default=2);args=p.parse_args()
    torch.set_num_threads(4);torch.manual_seed(2026);np.random.seed(2026);args.output.mkdir(exist_ok=True,parents=True)
    checkpoint=torch.load(args.checkpoint,map_location='cpu');actor=sac_actor(checkpoint,'cuda').eval()
    wm=world_model({'model':checkpoint['u0']['wm']},'cuda').eval();value=task_value({'value':checkpoint['u0']['value']},'cuda').eval()
    il=BCActor(len(wm.obs_mean),actor.fc_mean.out_features).cuda();il.load_state_dict(checkpoint['frozen_il']);il.eval()
    for model in [actor,wm,value,il]:model.requires_grad_(False)
    counter=InteractionCounter(args.output/'interactions.json')
    env=gym.make(checkpoint['config']['env_id'],num_envs=args.states,obs_mode='state',sim_backend='physx_cuda',
        control_mode=checkpoint['config']['control_mode'],reward_mode='sparse',reconfiguration_freq=1,
        max_episode_steps=args.prefix_steps+args.continuation_steps+1)
    prefix_env=CountedEnv(env,counter,'diagnostic_prefix')
    branch_envs=[CountedEnv(env,counter,name) for name in ['first_action_student','sequence_student']]
    with torch.no_grad():
        obs,_=env.reset(seed=30000);prefix_obs=[obs.clone()];prefix_actions=[]
        for _ in range(args.prefix_steps):
            action=actor.get_eval_action(obs);prefix_actions.append(action.clone());obs,*_=prefix_env.step(action);prefix_obs.append(obs.clone())
        reference=obs.clone();torch.manual_seed(2026)
        candidates=generate_candidates(reference,actor,wm,il);scored=score_candidates(candidates,actor,value)
        prefix_actions=torch.stack(prefix_actions);prefix_obs=torch.stack(prefix_obs)
        shape=(2,args.repeats,9,args.states)
        returns=np.zeros(shape);success=np.zeros(shape,dtype=bool);matching_error=np.zeros(shape)
        reward_traces=np.zeros(shape+(args.continuation_steps,),dtype=np.float32)
        early_rewards=np.zeros(shape+(2,));real_states=np.zeros(shape+(2,reference.shape[-1]),dtype=np.float32)
        real_terminal_value=np.zeros(shape)
        for mode in range(2):
            for repeat in range(args.repeats):
                for candidate in range(9):
                    obs,_=env.reset(seed=30000)
                    max_error=(obs-prefix_obs[0]).abs().amax(-1)
                    for t,action in enumerate(prefix_actions):
                        obs,*_=prefix_env.step(action)
                        max_error=torch.maximum(max_error,(obs-prefix_obs[t+1]).abs().amax(-1))
                    matching_error[mode,repeat,candidate]=max_error.cpu().numpy()
                    discount=1.
                    for t in range(args.continuation_steps):
                        action=candidates['actions'][:,candidate,t] if t<(1 if mode==0 else 2) else actor.get_eval_action(obs)
                        obs,reward,_,truncated,info=branch_envs[mode].step(action)
                        assert not truncated.any()
                        rewards=reward.cpu().numpy();assert np.isin(rewards,[0,1]).all()
                        reward_traces[mode,repeat,candidate,:,t]=rewards
                        returns[mode,repeat,candidate]+=discount*rewards;discount*=.99
                        success[mode,repeat,candidate]|=info['success'].cpu().numpy()
                        if t<2:
                            early_rewards[mode,repeat,candidate,:,t]=rewards
                            real_states[mode,repeat,candidate,:,t]=obs.cpu().numpy()
                        if t==1:real_terminal_value[mode,repeat,candidate]=value(obs,actor.get_eval_action(obs)).cpu().numpy()
                    counter.flush()
                    with (args.output/'branches.jsonl').open('a') as f:f.write(json.dumps(dict(mode=mode,repeat=repeat,candidate=candidate,
                        returns=returns[mode,repeat,candidate].tolist(),success=success[mode,repeat,candidate].tolist(),
                        prefix_max_error=matching_error[mode,repeat,candidate].tolist()))+'\n')
    env.close();counter.flush()
    np.testing.assert_allclose(returns,(reward_traces*.99**np.arange(args.continuation_steps)).sum(-1),atol=1e-6,rtol=1e-6)
    arrays=dict(returns=returns,reward_traces=reward_traces,success=success,matching_error=matching_error,early_rewards=early_rewards,real_states=real_states,
        real_terminal_value=real_terminal_value,prefix_observations=prefix_obs.cpu().numpy(),prefix_actions=prefix_actions.cpu().numpy(),
        **{k:v.cpu().numpy() for k,v in candidates.items() if isinstance(v,torch.Tensor)},
        **{k:v.cpu().numpy() for k,v in scored.items() if isinstance(v,torch.Tensor)})
    np.savez_compressed(args.output/'branches.npz',**arrays)
    selected=scored['selected'].cpu().numpy();predicted=scored['score'].cpu().numpy()
    valid=matching_error.max(axis=(0,1,2))<=1e-5
    tail_bound=.99**args.continuation_steps/(1-.99)
    summary={}
    for mode,name in enumerate(['first_action_student','sequence_student']):
        actual=returns[mode].mean(0).T
        rows=ranking_metrics(actual[valid],predicted[valid],selected[valid])
        summary[name]=dict(valid_state_indices=np.flatnonzero(valid).tolist(),states=rows,
            repeat_return_range_max=float(np.ptp(returns[mode],axis=0).max()),
            il_prediction_bias_to_finite_return=float((predicted[:,0]-actual[:,0])[valid].mean()) if valid.any() else None,
            sac_prediction_bias_to_finite_return=float((predicted[:,1:]-actual[:,1:])[valid].mean()) if valid.any() else None)
    actual_short=early_rewards[1,...,0]+.99*early_rewards[1,...,1]
    q_real=real_terminal_value[1]
    predicted_short=scored['short_return'].cpu().numpy().T[None]
    predicted_terminal=scored['terminal_component'].cpu().numpy().T[None]
    decomposition=dict(reward_error=predicted_short-actual_short,
        dynamics_terminal_error=predicted_terminal-.99**2*q_real,
        real_state_value_residual=.99**2*q_real-(returns[1]-actual_short))
    np.testing.assert_allclose(sum(decomposition.values()),predicted.T[None]-returns[1],atol=1e-5,rtol=1e-5)
    np.savez_compressed(args.output/'score_decomposition.npz',**decomposition,
        sequence_minus_first_action_return=returns[1]-returns[0])
    report=dict(checkpoint=str(args.checkpoint),checkpoint_sha256=hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),configuration=vars(args)|{'checkpoint':str(args.checkpoint),'output':str(args.output)},
        environment_time_limit=args.prefix_steps+args.continuation_steps+1,training_time_limit=checkpoint['config']['max_episode_steps'],
        time_limit_note='Artificial recording time limit extended only in diagnosis to measure continuing return without resetting; physics/control/sparse reward unchanged',
        matching_method='Repeated fresh resets at seed30000 and identical recorded deterministic student prefix; not arbitrary simulator-state restoration',
        matching_tolerance=1e-5,valid_states=int(valid.sum()),total_states=args.states,prefix_max_error=float(matching_error.max()),
        score_target='Infinite continuing task return after H2 commitment and deterministic student continuation',
        observed_target=f'Finite {args.continuation_steps}-step discounted real return, no reset during continuation',
        unobserved_tail_interval=[0,tail_bound],finite_return_tie_tolerance=1.,
        limitations='Eight matched states and repeated branches are not independent training runs. Unmatched states are retained in raw data but excluded from ranking. Pairwise metrics describe finite-horizon outcomes; differences below the remaining tail bound may change for infinite return. Repeat ranges are empirical spread, not confidence intervals.',
        new_training_interactions=0,results=summary)
    (args.output/'result.json').write_text(json.dumps(report,indent=2))


if __name__=='__main__':main()
