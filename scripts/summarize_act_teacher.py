"""Recompute fixed-final ACT evidence and freeze predeclared teacher selection (CPU)."""
import hashlib
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from scripts.act_teacher import ACTConfig, ACTTeacher
from scripts.train_act_teacher import write


def main():
    out = Path('artifacts/act-teacher-summary-001'); out.mkdir(exist_ok=False)
    rows = []
    for k, name in [(8, 'act-t0-seed0-001'), (16, 'act-t0-k16-seed0-001'), (30, 'act-t0-k30-seed0-001')]:
        p = Path('artifacts')/name
        result = json.loads((p/'result.json').read_text())
        manifest = json.loads((p/'manifest.json').read_text()); assert manifest['exit_code'] == 0
        episodes = [json.loads(s) for s in (p/'validation_episodes.jsonl').read_text().splitlines()]
        for field, method in [('final_validation','act_requery_first'), ('ensemble_validation','act_temporal_ensemble')]:
            selected = [r for r in episodes if r['iteration']==10000 and r['method']==method]
            assert len(selected)==40
            for metric in ['success_once', 'success_at_end']:
                assert sum(r[metric] for r in selected)==result[field][metric]
        resource = [json.loads(s) for s in (p/'resources.jsonl').read_text().splitlines()]
        rows.append(dict(k=k, experiment=name, result=result, gpu_job_seconds=manifest['elapsed_seconds'],
            peak_device_mib=max(r['memory_mib'] for r in resource),
            raw_episode_sha256=hashlib.sha256((p/'validation_episodes.jsonl').read_bytes()).hexdigest()))
    eligible = [r for r in rows if r['result']['teacher_gate_passed']]
    chosen = max(eligible, key=lambda r:(r['result']['final_validation']['success_once'],
        r['result']['final_validation']['success_at_end'], -r['k']))
    cp = Path('checkpoints')/chosen['experiment']/'final.pt'
    full = torch.load(cp, map_location='cpu', weights_only=False)
    model = ACTTeacher(full['state_dim'], full['action_dim'], ACTConfig(**full['model_config'])).eval()
    model.load_state_dict(full['model']); model.requires_grad_(False)
    # Portable model omits optimizer/RNG/replay; posterior tensors retained for strict loading.
    release = Path('releases/act-teacher-k30-001'); release.mkdir(exist_ok=False)
    weight = release/'weights.pt'
    torch.save({k:full[k] for k in ['model','model_config','state_dim','action_dim','step','config','selection']}, weight)
    reloaded = torch.load(weight, map_location='cpu', weights_only=False)
    assert all(torch.equal(v, reloaded['model'][n]) for n,v in full['model'].items())
    selection = dict(experiment=chosen['experiment'], chunk_size=chosen['k'], checkpoint=str(cp),
        checkpoint_sha256=hashlib.sha256(cp.read_bytes()).hexdigest(), portable=str(weight),
        portable_sha256=hashlib.sha256(weight.read_bytes()).hexdigest(), mode='current-state first action; latent=0; no temporal ensemble',
        normalization='none; raw state; action clamp [-1,1]', seed=10000,
        selection='fixed final; first-action success_once then success_at_end then smaller K',
        limitation='single development seed; 8/40 first-action successes, 4/40 end successes; usable threshold only')
    write(release/'manifest.json', selection)
    write(out/'summary.json', dict(conditions=rows, selection=selection, final_test_accessed=False))
    print(json.dumps(selection))

if __name__=='__main__': main()
