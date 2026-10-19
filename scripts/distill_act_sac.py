"""Frozen ACT first-action -> existing ManiSkill MLP-SAC; expert train states only."""
import argparse
import json
from pathlib import Path
import random
import sys
import time
import hashlib
from types import SimpleNamespace
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import gymnasium as gym
import numpy as np
import torch
from scripts.act_teacher import ACTConfig, ACTTeacher, EpisodeChunks
from scripts.train_sac import Actor, LOG_STD_MIN, LOG_STD_MAX
from scripts.policy_bridge import fingerprint, check_frozen
from scripts.train_act_teacher import write, append, closed_loop
from scripts.interaction_counter import InteractionCounter


def make_student(log_std=-2.0):
    env = SimpleNamespace(single_observation_space=gym.spaces.Box(-np.inf,np.inf,(42,),dtype=np.float32),
                          single_action_space=gym.spaces.Box(-1,1,(4,),dtype=np.float32))
    actor = Actor(env)
    with torch.no_grad():
        actor.fc_logstd.weight.zero_()
        actor.fc_logstd.bias.fill_(np.arctanh(2*(log_std-LOG_STD_MIN)/(LOG_STD_MAX-LOG_STD_MIN)-1))
    actor.fc_logstd.requires_grad_(False)
    return actor


@torch.no_grad()
def label(teacher, obs, batch=256):
    return torch.cat([teacher(x) for x in obs.split(batch)])


class ObservedPolicy:
    def __init__(self, student, teacher, stochastic):
        self.student, self.teacher, self.stochastic = student, teacher, stochastic
        self.squared = torch.zeros(4, device='cuda'); self.sample_squared = torch.zeros(4,device='cuda')
        self.states = []
        self.logprob = []
        self.count = 0

    def __call__(self, obs):
        target = self.teacher(obs)
        mean = self.student.get_eval_action(obs)
        if self.stochastic:
            action, lp, _ = self.student.get_action(obs)
            assert torch.isfinite(lp).all()
            self.logprob.append(lp.cpu().numpy())
        else:
            action = mean
        self.squared += (mean-target).square().sum(0)
        self.sample_squared += (action-mean).square().sum(0)
        self.count += len(obs)
        self.states.append(obs.cpu().numpy())
        return action


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--id',required=True)
    args=p.parse_args(); cfg=json.loads(args.config.read_text());out=args.output;out.mkdir(exist_ok=True)
    assert not (out/'train.jsonl').exists()
    torch.set_num_threads(2);torch.manual_seed(cfg['seed']);np.random.seed(cfg['seed']);random.seed(cfg['seed'])
    selection=json.loads(Path(cfg['selection']).read_text())
    checkpoint=Path(selection['portable']);assert hashlib.sha256(checkpoint.read_bytes()).hexdigest()==selection['portable_sha256']
    saved=torch.load(checkpoint,map_location='cpu',weights_only=False)
    teacher=ACTTeacher(saved['state_dim'],saved['action_dim'],ACTConfig(**saved['model_config'])).to('cuda')
    teacher.load_state_dict(saved['model']);teacher.eval().requires_grad_(False);before=fingerprint(teacher)
    split=json.loads(Path(cfg['audit']).read_text())
    assert split['alignment_validated'] and not set(split['train_episode_ids'])&set(split['validation_episode_ids'])
    assert hashlib.sha256(Path(split['dataset']).read_bytes()).hexdigest()==split['dataset_sha256']
    ds=EpisodeChunks.from_h5(split['dataset'],split['train_episode_ids'],1,'cuda')
    val=EpisodeChunks.from_h5(split['dataset'],split['validation_episode_ids'],1,'cuda')
    targets=label(teacher,ds.obs); val_targets=label(teacher,val.obs)
    student=make_student(cfg['log_std']).to('cuda');variance_before=fingerprint(student.fc_logstd)
    assert not {p.data_ptr() for p in student.parameters()}&{p.data_ptr() for p in teacher.parameters()}
    optimizer=torch.optim.Adam([p for p in student.parameters() if p.requires_grad],lr=cfg['lr'])
    write(out/'config.json',dict(**cfg,teacher=selection,train_episode_ids=split['train_episode_ids'],
        validation_episode_ids=split['validation_episode_ids'],dataset_sha256=split['dataset_sha256'],
        student_parameters=sum(p.numel() for p in student.parameters()),normalization='none',
        target='clip(ACT current-state zero-latent chunk[0], -1, 1); no ensemble',
        validation_labels_used_for_gradients=False,trained_parameters=[n for n,p in student.named_parameters() if p.requires_grad]))
    counter=InteractionCounter(out/'interactions.json');started=time.perf_counter()
    for step in range(1,cfg['updates']+1):
        idx=torch.randint(len(ds),(cfg['batch_size'],),device='cuda')
        loss=(student.get_eval_action(ds.obs[idx])-targets[idx]).square().mean()
        assert torch.isfinite(loss)
        optimizer.zero_grad();loss.backward();optimizer.step()
        append(out/'train.jsonl',dict(step=step,loss=float(loss.detach()),elapsed_seconds=time.perf_counter()-started))
    student.eval();check_frozen(teacher,optimizer,before)
    assert fingerprint(student.fc_logstd)==variance_before
    with torch.no_grad():
        metrics={}
        for name,obs,target in [('train',ds.obs,targets),('validation',val.obs,val_targets)]:
            pred=torch.cat([student.get_eval_action(x) for x in obs.split(256)])
            metrics[name]=dict(mse=float((pred-target).square().mean()),per_action_mse=(pred-target).square().mean(0).tolist())
        samples,lp,means=student.get_action(ds.obs[:256]);assert torch.isfinite(lp).all() and samples.abs().max()<=1
        _,std=student(ds.obs[:256]);assert torch.allclose(std,torch.full_like(std,cfg['log_std']))
    # Test standard temperature-loss gradients without modifying trained actor/temperature.
    alpha=torch.tensor(0.,device='cuda',requires_grad=True)
    alpha_loss=(-alpha.exp()*(lp.detach()-4)).mean();alpha_loss.backward()
    assert torch.isfinite(alpha.grad)
    final={}
    for stochastic in [False,True]:
        method='sac_stochastic' if stochastic else 'sac_mean'
        directory=out/method;directory.mkdir()
        torch.manual_seed(10000)
        policy=ObservedPolicy(student,teacher,stochastic)
        result=closed_loop(policy,directory,cfg['updates'],cfg['episodes'],counter,video=cfg['video'],method=method)
        result['inference_scope']='student action plus diagnostic teacher comparison/state logging; not pure student latency'
        result['teacher_action_mse_on_student_states']=(policy.squared/policy.count).tolist()
        result['sample_mean_action_mse']=(policy.sample_squared/policy.count).tolist()
        states=np.concatenate(policy.states)
        np.savez_compressed(directory/'state_distribution.npz',mean=states.mean(0),std=states.std(0),
            train_mean=ds.obs.mean(0).cpu().numpy(),train_std=ds.obs.std(0).cpu().numpy())
        if stochastic: np.savez_compressed(directory/'log_prob.npz',log_prob=np.concatenate(policy.logprob))
        final[method]=result
    # Same teacher states compare actions without environment-trajectory divergence.
    trace=np.load(cfg['teacher_trace'])['obs'].reshape(-1,42)
    with torch.no_grad():
        x=torch.as_tensor(trace,device='cuda');ref=label(teacher,x)
        pred=torch.cat([student.get_eval_action(b) for b in x.split(256)])
        np.savez_compressed(out/'paired_teacher_states.npz',teacher=ref.cpu().numpy(),student=pred.cpu().numpy())
        paired_mse=float((pred-ref).square().mean())
    check_frozen(teacher,optimizer,before)
    path=Path('checkpoints')/args.id/'final.pt';path.parent.mkdir(exist_ok=True,parents=True)
    torch.save(dict(actor=student.state_dict(),optimizer=optimizer.state_dict(),config=cfg,step=cfg['updates'],
        teacher_sha256=selection['portable_sha256'],rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all()),path)
    write(out/'checkpoints.json',dict(path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    write(out/'result.json',dict(offline=metrics,closed_loop=final,teacher_success_once=8,teacher_episodes=40,
        success_gap_teacher_minus_student={m:8/40-r['success_once']/r['episodes'] for m,r in final.items()},
        paired_teacher_state_mse=paired_mse,teacher_frozen=True,log_std_frozen=cfg['log_std'],
        temperature_gradient=float(alpha.grad),elapsed_seconds=time.perf_counter()-started,
        torch_peak_allocated_bytes=torch.cuda.max_memory_allocated(),new_training_interactions=0,
        final_test_accessed=False,smoke=cfg['updates']<10000))


if __name__=='__main__':main()
