"""Export a completed no-planning SAC actor without replay, optimizers or teacher."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from scripts.act_inference import load_model


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--checkpoint',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();torch.set_num_threads(2)
    full=torch.load(args.checkpoint,map_location='cpu',weights_only=False)
    config=full['config'];folder=Path(config['output'])
    run=json.loads((folder/'manifest.json').read_text())
    assert run['ended_at'] and run['exit_code']==0 and run['stop_reason'] is None
    assert config['env_id']=='PickCube-v1' and not config['wm_checkpoint']
    assert full['global_step']==config['total_timesteps']
    source_hash=hashlib.sha256(args.checkpoint.read_bytes()).hexdigest()
    records=[json.loads(line) for line in (folder/'checkpoints.jsonl').read_text().splitlines()]
    assert any(r['final'] and r['sha256']==source_hash for r in records)
    kind=config['actor_init_kind']
    package=dict(actor=full['actor'],actor_kind=kind,config=config,
                 training_interactions=full['lineage_training_interactions'],step=full['global_step'])
    if kind=='act':
        initialization=json.loads((folder/'actor_initialization.json').read_text())
        path=Path(config['actor_init_checkpoint'])
        assert hashlib.sha256(path.read_bytes()).hexdigest()==initialization['sha256']
        original=torch.load(path,map_location='cpu',weights_only=False)
        for key in ['model_config','state_dim','action_dim','log_std_init','action_epsilon']:
            package[key]=original[key]
        package['initialization_sha256']=initialization['sha256']
    else:
        assert kind=='mlp'
    model=load_model(package,kind)
    assert all(torch.equal(v,model.state_dict()[k]) for k,v in full['actor'].items())
    args.output.mkdir(parents=True,exist_ok=False)
    path=args.output/'weights.pt';torch.save(package,path)
    loaded=load_model(torch.load(path,map_location='cpu',weights_only=False),kind)
    assert all(torch.equal(v,loaded.state_dict()[k]) for k,v in full['actor'].items())
    torch.manual_seed(123);obs=torch.randn(32,42)
    with torch.no_grad():
        before=model.get_eval_action(obs);after=loaded.get_eval_action(obs)
        assert torch.equal(before,after)
    manifest=dict(source=str(args.checkpoint),source_sha256=source_hash,
        portable=str(path),portable_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),bytes=path.stat().st_size,
        actor_kind=kind,all_actor_tensors_equal=True,cpu_synthetic_action_parity=True,
        includes_replay=False,includes_optimizer=False,includes_teacher=False,new_environment_interactions=0,
        note='CPU serialization parity only; no new closed-loop performance or transfer evidence.')
    (args.output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps(manifest,indent=2))


if __name__=='__main__':main()
