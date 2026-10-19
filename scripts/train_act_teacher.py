"""ACT-T0: audited state demonstrations -> fixed-final, prior-inference teacher.

Adapted ManiSkill ACT AdamW/L1+KL/StepLR training, with our existing dataset and
evidence protocol. No environment/framework installation, EMA or score selection.
"""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import random
import sys
import time
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch

from scripts.act_teacher import ACTConfig, ACTTeacher, EpisodeChunks, TemporalEnsemble


def write(path, value):
    path.write_text(json.dumps(value, indent=2)+'\n')


def append(path, value):
    with path.open('a') as stream:
        stream.write(json.dumps(value)+'\n')


@torch.no_grad()
def offline_errors(model, dataset, batch_size):
    totals = dict(first_mse=0., chunk_l1=0., clipped_values=0, first_values=0, valid_values=0)
    for start in range(0, len(dataset), batch_size):
        indices = torch.arange(start, min(start+batch_size, len(dataset)), device=dataset.obs.device)
        obs, actions, pad = dataset.batch(indices)
        raw = model.raw_chunk(obs); pred = raw.clamp(-1, 1); valid = (~pad)[..., None]
        totals['first_mse'] += (pred[:, 0]-actions[:, 0]).square().sum().item()
        totals['chunk_l1'] += ((pred-actions).abs()*valid).sum().item()
        totals['clipped_values'] += ((raw.abs()>1)*valid).sum().item()
        totals['first_values'] += actions[:, 0].numel()
        totals['valid_values'] += int(valid.sum())*actions.shape[-1]
    return dict(first_action_mse=totals['first_mse']/totals['first_values'],
        prior_chunk_l1=totals['chunk_l1']/totals['valid_values'],
        raw_output_clip_fraction=totals['clipped_values']/totals['valid_values'], **totals)


@torch.no_grad()
def closed_loop(model, out, iteration, episodes, counter, video=False, temporal=False, method=None,
                matched_resets=True):
    from scripts.interaction_counter import CountedEnv
    from vendor.maniskill.bc.behavior_cloning.make_env import make_eval_envs
    method = method or ('act_temporal_ensemble' if temporal else 'act_requery_first')
    env = make_eval_envs('PickCube-v1', 8, 'physx_cuda', dict(obs_mode='state',
        control_mode='pd_ee_delta_pos', reward_mode='sparse', max_episode_steps=200,
        render_mode='rgb_array'), video_dir=str(out/('videos_ensemble' if temporal else 'videos')) if video else None,
        wrappers=[lambda e: CountedEnv(e, counter, 'validation')])
    assert episodes > 0 and episodes % 8 == 0
    reset_seed = 10000
    obs, _ = env.reset(seed=reset_seed)
    episode_count = 0; rows = []; latencies = []
    memory = TemporalEnsemble(8, model.config.num_queries, 4, 'cuda') if temporal else None
    events = {k: torch.zeros(8, dtype=torch.bool, device='cuda')
              for k in ['approached', 'is_grasped', 'lifted', 'is_obj_placed']}
    first_grasp = torch.full((8,), -1, dtype=torch.int64, device='cuda'); episode_step = 0
    traces = {k: [] for k in ['obs', 'action', 'reward']}
    while episode_count < episodes:
        if video: traces['obs'].append(obs.cpu().numpy())
        torch.cuda.synchronize(); before = time.perf_counter()
        action = memory.update(model.raw_chunk(obs)) if temporal else model(obs)
        torch.cuda.synchronize(); latencies.append((time.perf_counter()-before)*1000)
        obs, reward, _, truncated, info = env.step(action)
        # Final info belongs to the completed episode, not the auto-reset state.
        current = info['final_info'] if truncated.any() else info
        episode_step += 1
        first_grasp[(first_grasp<0) & current['is_grasped'].bool()] = episode_step
        events['is_grasped'] |= current['is_grasped'].bool()
        events['is_obj_placed'] |= current['is_obj_placed'].bool()
        physical_obs = info['final_observation'] if truncated.any() else obs
        events['approached'] |= physical_obs[:, -6:-3].norm(dim=-1) < .05
        # PickCube raw state layout: agent18 + grasp1 + TCP7 + goal3 + cube7 + offsets6.
        # Cube centre z > .04 means a .02m lift above its initial .02m table height.
        assert physical_obs.shape[-1] == 42
        events['lifted'] |= physical_obs[:, 31] > .04
        if video:
            traces['action'].append(action.cpu().numpy()); traces['reward'].append(reward.cpu().numpy())
        if truncated.any():
            assert truncated.all()
            metrics = {k: v.float().cpu().numpy() for k,v in current['episode'].items()}
            for i in range(8):
                row = dict(iteration=iteration, episode_index=episode_count+i, eval_seed=reset_seed,
                    method=method,
                    first_grasp_step=int(first_grasp[i]) if first_grasp[i]>=0 else None,
                    **{k: float(v[i]) for k,v in metrics.items()},
                    **{k+'_once': bool(v[i]) for k,v in events.items()})
                rows.append(row); append(out/'validation_episodes.jsonl', row)
            for event in events.values(): event.zero_()
            first_grasp.fill_(-1); episode_step = 0
            if memory is not None: memory.reset(truncated)
            episode_count += 8
            if matched_resets and episode_count < episodes:
                # Unseeded auto-resets consume global Torch RNG, including policy draws.
                # Discard that reset observation and explicitly seed the next full batch.
                reset_seed = 10000 + episode_count
                obs, _ = env.reset(seed=reset_seed)
    env.close(); counter.flush()
    append(out/'inference_latency.jsonl', dict(iteration=iteration, method=method,
        batch_size=8, milliseconds=latencies))
    if video: np.savez_compressed(out/('final_ensemble_trace.npz' if temporal else 'final_validation_trace.npz'),
        **{k: np.stack(v) for k,v in traces.items()})
    return dict(method=method, episodes=len(rows), success_once=sum(r['success_once'] for r in rows),
        success_at_end=sum(r['success_at_end'] for r in rows),
        mean_return=float(np.mean([r['return'] for r in rows])),
        stage_counts={k: sum(r[k+'_once'] for r in rows) for k in events},
        latency_batch8_ms_mean=float(np.mean(latencies)), latency_batch8_ms_p95=float(np.quantile(latencies,.95)),
        inference_scope=('prior full chunk + temporal aggregation + clamp' if temporal else
                         method)+'; synchronized CUDA; excludes env.step/render',
        validation_seed=10000, reset_protocol='explicit seed 10000+episode_batch_start' if matched_resets else 'legacy initial seed only')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--config', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--id', required=True)
    p.add_argument('--cpu-audit', action='store_true')
    args = p.parse_args(); cfg = json.loads(args.config.read_text())
    out = args.output; out.mkdir(parents=True, exist_ok=True)
    assert not (out/'train.jsonl').exists(), 'Fresh experiment ID required'
    torch.set_num_threads(2); random.seed(cfg['seed']); np.random.seed(cfg['seed']); torch.manual_seed(cfg['seed'])
    device = torch.device('cpu' if args.cpu_audit else 'cuda')
    split_path = Path(cfg['audit']); split = json.loads(split_path.read_text())
    assert split['alignment_validated']
    assert not (split_path.parent/'INVALIDATED.json').exists()
    assert not (split_path.parent/'INITIAL_OBSERVATION_CAVEAT.json').exists()
    assert not set(split['train_episode_ids']) & set(split['validation_episode_ids'])
    assert hashlib.sha256(Path(split['dataset']).read_bytes()).hexdigest() == split['dataset_sha256']
    model_cfg = ACTConfig(**cfg['model'])
    ds = EpisodeChunks.from_h5(split['dataset'], split['train_episode_ids'], model_cfg.num_queries, device)
    val = EpisodeChunks.from_h5(split['dataset'], split['validation_episode_ids'], model_cfg.num_queries, device)
    assert split['env_info']['env_id'] == 'PickCube-v1'
    assert split['env_info']['env_kwargs']['control_mode'] == 'pd_ee_delta_pos'
    assert split['env_info']['env_kwargs']['reward_mode'] == 'sparse'
    for dataset in (ds, val):
        assert dataset.obs.shape[1] == 42 and dataset.actions.shape[1] == 4
        assert torch.isfinite(dataset.obs).all() and torch.isfinite(dataset.actions).all()
        assert dataset.actions.abs().max() <= 1
    model = ACTTeacher(ds.obs.shape[-1], ds.actions.shape[-1], model_cfg).to(device)
    params = sum(p.numel() for p in model.parameters()); assert params < 10_000_000
    posterior_prefixes = ('encoder.', 'cls_embed.', 'encoder_state_proj.', 'encoder_action_proj.', 'latent_proj.')
    posterior_params = sum(p.numel() for n,p in model.model.named_parameters() if n.startswith(posterior_prefixes))
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg['lr'], weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, max(1, int(cfg['updates']*2/3)), gamma=.1)
    write(out/'config.json', dict(**cfg, cpu_audit=args.cpu_audit, model_parameters=params,
        posterior_only_parameters=posterior_params, inference_parameters=params-posterior_params,
        dataset_sha256=split['dataset_sha256'], train_episode_ids=split['train_episode_ids'],
        validation_episode_ids=split['validation_episode_ids'], normalization='none',
        posterior_padding='masked', loss=f'valid-action L1 + {model_cfg.kl_weight}*KL', ema=False,
        evaluation_protocol=dict(env_id='PickCube-v1', obs_mode='state', reward_mode='sparse',
            control_mode='pd_ee_delta_pos', sim_backend='physx_cuda', reconfiguration_freq=1,
            max_episode_steps=200, num_envs=8, seed=10000, ignore_terminations=True),
        decoder_output='final layer; upstream intermediate[0] corrected'))
    counter = None
    if not args.cpu_audit:
        from scripts.interaction_counter import InteractionCounter
        counter = InteractionCounter(out/'interactions.json')
    started = time.perf_counter(); final_validation = None; ensemble_validation = None
    for step in range(1, cfg['updates']+1):
        # Uniform transition sampling with replacement; explicit deviation from official epoch sampler.
        batch = ds.batch(torch.randint(len(ds), (cfg['batch_size'],), device=device))
        model.train(); loss = model.loss(*batch)
        assert torch.isfinite(loss['loss']), 'Nonfinite ACT objective'
        optimizer.zero_grad(); loss['loss'].backward(); optimizer.step(); scheduler.step()
        append(out/'train.jsonl', dict(step=step, **{k: float(v.detach()) for k,v in loss.items()},
            learning_rate=optimizer.param_groups[0]['lr'], elapsed_seconds=time.perf_counter()-started))
        if step % cfg['eval_every'] == 0 or step == cfg['updates']:
            model.eval()
            rng = (random.getstate(), np.random.get_state(), torch.get_rng_state(),
                   torch.cuda.get_rng_state_all() if device.type=='cuda' else None)
            errors = offline_errors(model, val, cfg['batch_size'])
            append(out/'offline_validation.jsonl', dict(step=step, **errors))
            if not args.cpu_audit:
                final_validation = closed_loop(model, out, step, cfg['validation_episodes'], counter,
                    video=step==cfg['updates'] and cfg['record_final_video'])
                append(out/'validation_summary.jsonl', dict(step=step, **final_validation))
                if step==cfg['updates'] and cfg['evaluate_temporal_ensemble']:
                    ensemble_validation = closed_loop(model, out, step, cfg['validation_episodes'], counter,
                        video=cfg['record_final_video'], temporal=True)
                    append(out/'validation_summary.jsonl', dict(step=step, **ensemble_validation))
            random.setstate(rng[0]); np.random.set_state(rng[1]); torch.set_rng_state(rng[2])
            if rng[3] is not None: torch.cuda.set_rng_state_all(rng[3])
            directory = Path('checkpoints')/args.id; directory.mkdir(parents=True, exist_ok=True)
            path = directory/('final.pt' if step==cfg['updates'] else f'step-{step}.pt')
            torch.save(dict(model=model.state_dict(), model_config=asdict(model_cfg),
                state_dim=ds.obs.shape[-1], action_dim=ds.actions.shape[-1], optimizer=optimizer.state_dict(),
                scheduler=scheduler.state_dict(), step=step, config=cfg, rng=rng,
                selection='fixed_final' if step==cfg['updates'] else 'periodic_not_selected'), path)
            append(out/'checkpoints.jsonl', dict(path=str(path), step=step, bytes=path.stat().st_size,
                sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
            print(json.dumps(dict(step=step, offline=errors, closed_loop=final_validation)), flush=True)
    write(out/'result.json', dict(updates=cfg['updates'], elapsed_seconds=time.perf_counter()-started,
        model_parameters=params, training_examples=len(ds), validation_examples=len(val),
        final_validation=final_validation, ensemble_validation=ensemble_validation,
        posterior_only_parameters=posterior_params, inference_parameters=params-posterior_params,
        final_test_accessed=False, cpu_audit=args.cpu_audit,
        new_training_environment_interactions=0, teacher_gate_passed=(final_validation is not None and
            final_validation['episodes']==40 and final_validation['success_once']>=8),
        gate_definition='Fixed final, mode A, >=8/40 validation success before migration; not significance or 3-seed confirmation',
        torch_peak_allocated_bytes=torch.cuda.max_memory_allocated() if device.type=='cuda' else 0))


if __name__ == '__main__':
    main()
