"""U2: finite expert-state mean retention; no teacher in SAC updates/deployment."""
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch


class MeanRetention:
    def __init__(self, observations, targets, weight, end_step, batch_size, seed):
        assert weight > 0 and end_step > 0 and batch_size > 0
        assert len(observations) == len(targets) > 0
        self.observations = observations.detach()
        self.targets = targets.detach()
        self.weight, self.end_step, self.batch_size = weight, end_step, batch_size
        self.generator = torch.Generator().manual_seed(seed)
        self.updates = self.samples = 0

    def coefficient(self, step):
        return self.weight * max(0., 1. - step / self.end_step)

    def loss(self, actor, step):
        coefficient = self.coefficient(step)
        if coefficient == 0:
            return self.targets.new_zeros(()), coefficient
        indices = torch.randint(len(self.targets), (self.batch_size,), generator=self.generator)
        mean = actor.get_eval_action(self.observations[indices])
        mse = (mean - self.targets[indices]).square().mean()
        self.updates += 1
        self.samples += self.batch_size
        return mse, coefficient

    @classmethod
    def from_config(cls, config_path, device, output):
        from scripts.act_inference import load_model
        from scripts.act_teacher import EpisodeChunks
        from scripts.policy_bridge import fingerprint
        started = time.perf_counter()
        cfg = json.loads(Path(config_path).read_text())
        audit = json.loads(Path(cfg['audit']).read_text())
        assert audit['alignment_validated']
        assert not set(audit['train_episode_ids']) & set(audit['validation_episode_ids'])
        assert hashlib.sha256(Path(audit['dataset']).read_bytes()).hexdigest() == audit['dataset_sha256']
        teacher_path = Path(cfg['teacher'])
        assert hashlib.sha256(teacher_path.read_bytes()).hexdigest() == cfg['teacher_sha256']
        # Model construction and label generation must not perturb rollout/SAC RNG.
        with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
            teacher = load_model(torch.load(teacher_path, map_location='cpu', weights_only=False), 'teacher', device)
            before = fingerprint(teacher)
            data = EpisodeChunks.from_h5(audit['dataset'], audit['train_episode_ids'], 1, device)
            with torch.no_grad():
                targets = torch.cat([teacher(batch) for batch in data.obs.split(256)])
            assert fingerprint(teacher) == before and not any(p.requires_grad for p in teacher.parameters())
            assert torch.isfinite(targets).all() and targets.abs().max() <= 1
            del teacher
        labels = output / 'retention_labels.npz'
        np.savez_compressed(labels, targets=targets.cpu().numpy())
        evidence = dict(config=cfg, dataset_sha256=audit['dataset_sha256'],
            train_episode_ids=audit['train_episode_ids'], states=len(data.obs),
            teacher_frozen_verified=True, labels_sha256=hashlib.sha256(labels.read_bytes()).hexdigest(),
            target='current-state latent=0 clipped chunk[0]; no history/ensemble',
            label_order='train episode order, pre-action observations; terminal observation excluded',
            teacher_calls_in_sac_updates=False, expert_transitions_in_critic=False,
            initialization_wall_seconds=time.perf_counter()-started)
        (output/'retention_manifest.json').write_text(json.dumps(evidence, indent=2)+'\n')
        return cls(data.obs, targets, cfg['weight'], cfg['end_step'], cfg['batch_size'], cfg['sampling_seed'])
