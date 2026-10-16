"""U0-only planning/model additions called by the official SAC training loop."""
import json
import time
from pathlib import Path
import torch
from scripts.load_components import world_model,task_value
from scripts.task_value import make_target,update_value
from scripts.planning import generate_candidates,score_candidates
from scripts.sac_evidence import append_json,evaluate_policy


class InferencePolicy:
    def __init__(self,fn):self.get_eval_action=fn
    def eval(self):return self
    def train(self):return self


class U0:
    def __init__(self,args,actor,il,expert,device,output):
        assert il is not None and expert is not None and args.gamma==.99
        self.actor,self.il,self.expert,self.output=actor,il,expert,output
        wm_state=torch.load(args.wm_checkpoint,map_location='cpu')
        value_state=torch.load(args.task_value_checkpoint,map_location='cpu')
        self.wm=world_model(wm_state,device);self.value=task_value(value_state,device)
        self.target=make_target(self.value);self.target.load_state_dict(value_state['target'])
        self.wm_optimizer=torch.optim.Adam(self.wm.parameters(),lr=3e-4);self.wm_optimizer.load_state_dict(wm_state['optimizer'])
        self.value_optimizer=torch.optim.Adam(self.value.parameters(),lr=3e-4);self.value_optimizer.load_state_dict(value_state['optimizer'])
        il_ids={id(p) for p in il.parameters()}
        for opt in [self.wm_optimizer,self.value_optimizer]:
            assert not il_ids&{id(p) for group in opt.param_groups for p in group['params']}
        self.wm_updates=0;self.value_updates=0
        self.counts=torch.zeros(3,dtype=torch.int64,device=device) # actual IL / planned SAC / random SAC
        self.chosen=torch.zeros(9,dtype=torch.int64,device=device)
        (output/'u0_config.json').write_text(json.dumps(dict(horizon=2,il_candidates=1,sac_candidates=8,no_il_sac_candidates=9,
            exploration_probability=.2,wm_update_every_sac_updates=4,wm_batch_max=512,value_update_every_sac_updates=1,
            wm_checkpoint=args.wm_checkpoint,task_value_checkpoint=args.task_value_checkpoint,
            objective='pure task return, deterministic current student continuation',wm_normalization='fixed preheat training statistics'),indent=2))

    @torch.no_grad()
    def act(self,obs,step):
        measured=step%1024==0
        if measured:torch.cuda.synchronize();start=time.perf_counter()
        candidates=generate_candidates(obs,self.actor,self.wm,self.il)
        result=score_candidates(candidates,self.actor,self.value)
        explore=torch.rand(len(obs),device=obs.device)<.2
        action=torch.where(explore[:,None],self.actor.get_action(obs)[0],result['action'])
        source=torch.where(explore,2,torch.where(result['selected']==0,0,1)).to(torch.int8)
        self.counts+=torch.bincount(source.long(),minlength=3)
        self.chosen+=torch.bincount(result['selected'],minlength=9)
        if measured:
            torch.cuda.synchronize();latency=(time.perf_counter()-start)*1000
            assert torch.isfinite(result['score']).all()
            append_json(self.output/'planning.jsonl',dict(step=step,latency_ms=latency,batch_size=len(obs),
                selected=result['selected'].cpu().tolist(),executed_source=source.cpu().tolist(),
                score=result['score'].cpu().tolist(),short_return=result['short_return'].cpu().tolist(),
                terminal_component=result['terminal_component'].cpu().tolist(),executed_action=action.cpu().tolist()))
        return action,source,result['selected']

    def update(self,data,replay,update):
        self.value_loss=update_value(self.value,self.target,self.value_optimizer,self.actor,data.obs,data.actions,data.rewards,data.next_obs)
        self.value_updates+=1
        if update%4==0:
            online=replay.sample_sequences(256)
            n=len(online['actions'])
            if n:
                expert=self.expert.sample_sequences(n)
                batch={k:torch.cat([online[k],expert[k]]) for k in online}
                loss,_=self.wm.loss(**batch)
                self.wm_optimizer.zero_grad();loss.backward();self.wm_optimizer.step()
                self.wm_loss=loss.detach();self.wm_updates+=1

    def evaluate(self,env,args,step):
        policies={'il':InferencePolicy(self.il),'sac':self.actor}
        for name,teacher in [('il_sac_wm',self.il),('sac_wm',None)]:
            policies[name]=InferencePolicy(lambda obs,teacher=teacher:score_candidates(generate_candidates(obs,self.actor,self.wm,teacher),self.actor,self.value)['action'])
        count=0
        for name,policy in policies.items():
            start=time.perf_counter()
            count+=evaluate_policy(policy,env,args,self.output,step,method=name)
            append_json(self.output/'evaluation_cost.jsonl',dict(step=step,method=name,wall_seconds=time.perf_counter()-start,
                candidates=9 if name.endswith('wm') else 1,horizon=2 if name.endswith('wm') else 1,
                environment_interactions=args.num_eval_envs*args.num_eval_steps))
        return count

    def restore(self,state):
        self.wm.load_state_dict(state['wm']);self.value.load_state_dict(state['value']);self.target.load_state_dict(state['value_target'])
        self.wm_optimizer.load_state_dict(state['wm_optimizer']);self.value_optimizer.load_state_dict(state['value_optimizer'])
        (self.output/'u0_parent_counts.json').write_text(json.dumps(dict(wm_updates=state['wm_updates'],value_updates=state['value_updates'],
            executed_sources=state['executed_source_counts'].tolist(),new_segment_counters_start_at_zero=True),indent=2))

    def log(self,step):
        append_json(self.output/'model_updates.jsonl',dict(step=step,wm_updates=self.wm_updates,value_updates=self.value_updates,
            wm_loss=float(self.wm_loss) if self.wm_updates else None,value_loss=float(self.value_loss) if self.value_updates else None,
            executed_il=int(self.counts[0]),executed_planned_sac=int(self.counts[1]),executed_random_sac=int(self.counts[2])))

    def snapshot(self):
        return dict(wm=self.wm.state_dict(),wm_optimizer=self.wm_optimizer.state_dict(),value=self.value.state_dict(),
            value_target=self.target.state_dict(),value_optimizer=self.value_optimizer.state_dict(),
            wm_updates=self.wm_updates,value_updates=self.value_updates,executed_source_counts=self.counts.cpu(),chosen_candidate_counts=self.chosen.cpu())

    @property
    def parameters_count(self):return sum(p.numel() for m in [self.wm,self.value] for p in m.parameters())
