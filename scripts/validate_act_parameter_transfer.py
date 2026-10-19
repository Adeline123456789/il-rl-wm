"""C1: parameter audit, real-state stochastic gradients, matched closed-loop test."""
import copy
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys
import time
from types import SimpleNamespace
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import gymnasium as gym
import numpy as np
import torch
from scripts.act_teacher import ACTConfig,ACTTeacher,EpisodeChunks
from scripts.act_sac_actor import ACTSACActor
from scripts.train_sac import SoftQNetwork
from scripts.policy_bridge import fingerprint
from scripts.train_act_teacher import closed_loop,write
from scripts.interaction_counter import InteractionCounter


def main():
    torch.set_num_threads(2);torch.manual_seed(0)
    out=Path('artifacts/act-c1-seed0-001');started=time.perf_counter()
    selected=json.loads(Path('releases/act-teacher-k30-001/manifest.json').read_text())
    path=Path(selected['portable']);assert hashlib.sha256(path.read_bytes()).hexdigest()==selected['portable_sha256']
    full=torch.load(path,map_location='cpu',weights_only=False)
    teacher=ACTTeacher(42,4,ACTConfig(**full['model_config'])).eval().requires_grad_(False)
    teacher.load_state_dict(full['model']);teacher_hash=fingerprint(teacher)
    actor=ACTSACActor.from_teacher(teacher).to('cuda');teacher.to('cuda')
    copies=actor.copy_audit(teacher);write(out/'copied_parameters.json',copies)
    cfg=dict(teacher=selected,model_config=full['model_config'],state_dim=42,action_dim=4,
        mapping='pre_tanh_mean=atanh(clamp(copied_action_head(h0),-1+1e-6,1-1e-6))',
        log_std_init=-2,action_epsilon=1e-6,dropout='off during all inference and SAC gradient updates',
        posterior_encoder='not constructed/copied',queries='all 30 before first-query output',
        critic_probe='random frozen twin LayerNorm Q; engineering only, no RL training',
        comparison='fixed final K30 and original D1; explicit matched resets; no model selection',
        actual_actor_parameters=sum(p.numel() for p in actor.parameters()),
        copied_inference_parameters=sum(p.numel() for n,p in actor.named_parameters() if n.split('.')[0] in actor.copied_modules),
        new_variance_parameters=sum(p.numel() for p in actor.fc_logstd.parameters()))
    write(out/'config.json',cfg)
    split=json.loads(Path('artifacts/e0-pickcube-gpu-replay-full-003/dataset-audit.json').read_text())
    assert hashlib.sha256(Path(split['dataset']).read_bytes()).hexdigest()==split['dataset_sha256']
    ds=EpisodeChunks.from_h5(split['dataset'],split['train_episode_ids'],1,'cuda')
    max_error=0;sum_error=0;saturated=0;count=0
    with torch.no_grad():
        for obs in ds.obs.split(256):
            target=teacher(obs);pred=actor.get_eval_action(obs)
            max_error=max(max_error,float((pred-target).abs().max()));sum_error+=float((pred-target).square().sum())
            saturated+=int((target.abs()>=1-actor.action_epsilon).sum());count+=target.numel()
    assert max_error<5e-5
    # Independent copy verifies actual pathwise actor/temperature derivatives on real states.
    probe=copy.deepcopy(actor);obs=ds.obs[:256]
    env=SimpleNamespace(single_observation_space=gym.spaces.Box(-np.inf,np.inf,(42,),dtype=np.float32),single_action_space=gym.spaces.Box(-1,1,(4,),dtype=np.float32))
    critics=[SoftQNetwork(env,layer_norm=True).to('cuda').requires_grad_(False) for _ in range(2)]
    optimizer=torch.optim.Adam(probe.parameters(),lr=3e-4)
    action,lp,_=probe.get_action(obs);assert torch.isfinite(lp).all()
    loss=(.2*lp-torch.minimum(*[q(obs,action) for q in critics])).mean()
    optimizer.zero_grad();loss.backward()
    gradients={n:float(p.grad.norm()) for n,p in probe.named_parameters() if p.grad is not None}
    assert all(np.isfinite(list(gradients.values()))) and gradients['fc_logstd.weight']>0
    assert gradients['transformer.decoder.layers.1.linear1.weight']>0
    optimizer.step()
    alpha=torch.tensor(0.,device='cuda',requires_grad=True)
    (-alpha.exp()*(lp.detach()-4)).mean().backward();assert torch.isfinite(alpha.grad)
    write(out/'gradient_probe.json',dict(loss=float(loss.detach()),gradient_norms=gradients,alpha_gradient=float(alpha.grad),
        changes_evaluated_actor=False,training_interactions=0,warning='Random critic gradient plumbing only; not learning or stability evidence'))
    # Measure uninstrumented action calls on real states, excluding env.step.
    latencies=[]
    with torch.no_grad():
        for i in range(110):
            torch.cuda.synchronize();t=time.perf_counter();actor.get_eval_action(ds.obs[:8]);torch.cuda.synchronize()
            if i>=10:latencies.append((time.perf_counter()-t)*1000)
    write(out/'inference_latency.json',dict(batch_size=8,milliseconds=latencies,mean=float(np.mean(latencies)),p95=float(np.quantile(latencies,.95))))
    counter=InteractionCounter(out/'interactions.json');actor.eval();results={}
    reference=np.load('artifacts/act-matched-eval-001/act_k30_first/final_validation_trace.npz')['obs'][::200]
    for stochastic in [False,True]:
        name='transfer_stochastic' if stochastic else 'transfer_mean';directory=out/name;directory.mkdir()
        torch.manual_seed(10000)
        policy=(lambda x:actor.get_action(x)[0]) if stochastic else actor.get_eval_action
        r=closed_loop(policy,directory,0,40,counter,video=True,method=name,matched_resets=True)
        initial=np.load(directory/'final_validation_trace.npz')['obs'][::200]
        r['initial_obs_max_abs_vs_teacher']=float(abs(initial-reference).max());assert r['initial_obs_max_abs_vs_teacher']<1e-4
        results[name]=r
    assert fingerprint(teacher)==teacher_hash
    actor.copy_audit(teacher) # gradient probe was separate; evaluation must not modify copied weights
    release=Path('releases/act-c1-init-001');release.mkdir(exist_ok=False)
    weight=release/'weights.pt';torch.save(dict(actor=actor.state_dict(),model_config=asdict(actor.config),state_dim=42,action_dim=4,action_epsilon=1e-6,log_std_init=-2),weight)
    write(release/'manifest.json',dict(source_teacher=selected,sha256=hashlib.sha256(weight.read_bytes()).hexdigest(),
        teacher_required_at_deployment=False,requires_class='scripts.act_sac_actor.ACTSACActor',online_training_completed=False))
    write(out/'result.json',dict(real_expert_states=len(ds),initial_action_max_abs=max_error,initial_action_mse=sum_error/count,
        saturated_action_fraction=saturated/count,copied_tensor_count=len(copies),teacher_unchanged=True,
        methods=results,elapsed_seconds=time.perf_counter()-started,torch_peak_allocated_bytes=torch.cuda.max_memory_allocated(),
        train_interactions=0,final_test_accessed=False,actor_parameters=cfg['actual_actor_parameters'],
        online_learning_verified=False,note='Initial parameter preservation and stochastic execution only; online SAC superiority/forgetting untested'))

if __name__=='__main__':main()
