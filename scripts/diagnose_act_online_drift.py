"""CPU fixed-input ACT transfer drift; no new data, updates or simulation."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import h5py
import numpy as np
import torch
from scripts.act_inference import load_model


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def train_probe(audit_path, count=1024):
    audit = json.loads(audit_path.read_text())
    assert audit['alignment_validated']
    assert not set(audit['train_episode_ids']) & set(audit['validation_episode_ids'])
    assert digest(audit['dataset']) == audit['dataset_sha256']
    observations, ids, times = [], [], []
    remaining = count
    with h5py.File(audit['dataset']) as source:
        for episode in audit['train_episode_ids']:
            trajectory = source[f'traj_{episode}']
            assert len(trajectory['obs']) == len(trajectory['actions']) + 1
            n = min(len(trajectory['actions']), remaining)
            observations.append(trajectory['obs'][:n])
            ids.extend([episode] * n); times.extend(range(n)); remaining -= n
            if remaining == 0:
                break
    assert remaining == 0
    obs = torch.as_tensor(np.concatenate(observations), dtype=torch.float32)
    assert obs.shape == (count, 42) and torch.isfinite(obs).all()
    return obs, np.asarray(ids), np.asarray(times), audit


@torch.no_grad()
def probe(actor, observations):
    means, logs = [], []
    for batch in observations.split(64):
        mean, log_std = actor(batch)
        means.append(mean); logs.append(log_std)
    mean = torch.cat(means); log_std = torch.cat(logs)
    assert torch.isfinite(mean).all() and torch.isfinite(log_std).all()
    return dict(pre_tanh_mean=mean, log_std=log_std,
                action=mean.tanh()*actor.action_scale+actor.action_bias)


def gaussian_kl(initial, current):
    """KL(initial Gaussian || current Gaussian), before the common tanh map."""
    m0, l0 = initial['pre_tanh_mean'], initial['log_std']
    m1, l1 = current['pre_tanh_mean'], current['log_std']
    return (l1-l0 + .5*((2*(l0-l1)).exp() + (m0-m1).square()*(-2*l1).exp() - 1)).sum(-1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment', required=True)
    parser.add_argument('--steps', nargs='+', type=int, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(); torch.set_num_threads(2)
    folder = Path('artifacts') / args.experiment
    manifest = json.loads((folder/'manifest.json').read_text())
    assert manifest['ended_at'] and manifest['exit_code'] == 0 and manifest['stop_reason'] is None
    config = json.loads((folder/'sac_config.json').read_text())
    assert config['env_id'] == 'PickCube-v1' and config['eval_seed'] == 10000
    assert config['actor_init_checkpoint'] and not config['wm_checkpoint']
    assert args.steps == sorted(set(args.steps)) and args.steps[0] > 0
    init_path = Path(config['actor_init_checkpoint'])
    initialization = json.loads((folder/'actor_initialization.json').read_text())
    assert digest(init_path) == initialization['sha256']
    initial = torch.load(init_path, map_location='cpu', weights_only=False)
    kind = config['actor_init_kind']
    actor = load_model(initial, kind)
    # All registered transfer runs reset to this same variance before online RL.
    assert config['initial_log_std'] == -2
    obs, ids, times, audit = train_probe(Path('artifacts/e0-pickcube-gpu-replay-full-003/dataset-audit.json'))
    baseline = probe(actor, obs)
    assert torch.allclose(baseline['log_std'], torch.full_like(baseline['log_std'], -2), atol=1e-6)
    teacher_path = Path('releases/act-teacher-k30-001/weights.pt')
    selection = json.loads((teacher_path.parent/'manifest.json').read_text())
    assert digest(teacher_path) == selection['portable_sha256']
    teacher = load_model(torch.load(teacher_path, map_location='cpu', weights_only=False), 'teacher')
    with torch.no_grad():
        target = torch.cat([teacher(batch) for batch in obs.split(64)])
    arrays = dict(observations=obs.numpy(), episode_ids=ids, timestep=times, teacher_action=target.numpy())
    results = []
    records = [json.loads(line) for line in (folder/'checkpoints.jsonl').read_text().splitlines()]
    episodes = [json.loads(line) for line in (folder/'validation_episodes.jsonl').read_text().splitlines()]
    for step in [0, *args.steps]:
        source = dict(path=str(init_path), sha256=digest(init_path))
        weights = initial['actor']
        if step:
            matches = [r for r in records if r['step'] == step]
            assert len(matches) == 1
            source = matches[0]
            assert digest(source['path']) == source['sha256']
            checkpoint = torch.load(source['path'], map_location='cpu', weights_only=False)
            assert checkpoint['global_step'] == step and checkpoint['config'] == config
            weights = checkpoint['actor']
            actor.load_state_dict(weights, strict=True)
        current = probe(actor, obs)
        for key, value in current.items():
            arrays[f'step{step}_{key}'] = value.numpy()
        kl = gaussian_kl(baseline, current)
        arrays[f'step{step}_kl_initial_current'] = kl.numpy()
        validation = [row for row in episodes if row['step'] == step]
        assert len(validation) == 40
        delta = {name: float((weights[name]-value).norm()) for name, value in initial['actor'].items()}
        results.append(dict(step=step, checkpoint=source,
            teacher_action_mse_per_dim=(current['action']-target).square().mean(0).tolist(),
            initial_action_drift_mse_per_dim=(current['action']-baseline['action']).square().mean(0).tolist(),
            mean_log_std=current['log_std'].mean(0).tolist(),
            mean_action_abs_max=current['action'].abs().amax(0).tolist(),
            near_boundary_action_fraction=(current['action'].abs()>=.999).float().mean(0).tolist(),
            pre_tanh_kl_initial_current_mean=float(kl.mean()), parameter_delta_l2=delta,
            validation_success_once=int(sum(row['success_once'] for row in validation)),
            validation_success_at_end=int(sum(row['success_at_end'] for row in validation)),
            validation_episodes=len(validation)))
        if step:
            del checkpoint
    args.output.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(args.output/'arrays.npz', **arrays)
    result = dict(experiment=args.experiment, actor_kind=kind, fixed_states=len(obs),
        state_selection='first 1024 expert-training transitions in registered episode order; same as smoke probe',
        dataset_sha256=audit['dataset_sha256'], initialization_sha256=initialization['sha256'],
        teacher_sha256=selection['portable_sha256'], results=results,
        new_environment_interactions=0, final_test_accessed=False,
        interpretation='Fixed expert-input drift plus existing validation. No online-distribution coverage, '
                       'causal identification, or new closed-loop evaluation. Gaussian KL is pre-tanh, not task value.')
    (args.output/'result.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps([{k:v for k,v in row.items() if k != 'parameter_delta_l2'} for row in results], indent=2))


if __name__ == '__main__':
    main()
