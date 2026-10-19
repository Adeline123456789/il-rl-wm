"""CPU diagnostic of T5 initial/8192 actor drift on fixed expert-train states."""
import hashlib,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import h5py
import numpy as np
import torch
from scripts.act_sac_actor import ACTSACActor
from scripts.act_teacher import ACTConfig
from scripts.train_act_teacher import write


def main():
    torch.set_num_threads(2)
    out=Path('artifacts/act-t5-smoke-diagnosis-001');out.mkdir(exist_ok=False)
    initial=torch.load('releases/act-c1-init-001/weights.pt',map_location='cpu',weights_only=False)
    trained=torch.load('checkpoints/act-t5-online-smoke-001/final.pt',map_location='cpu',weights_only=False)
    split=json.loads(Path('artifacts/e0-pickcube-gpu-replay-full-003/dataset-audit.json').read_text())
    with h5py.File(split['dataset']) as f:
        arrays=[f[f'traj_{i}/obs'][:-1] for i in split['train_episode_ids'][:20]]
    obs=torch.as_tensor(np.concatenate(arrays)[:1024],dtype=torch.float32)
    states={};preds={};raws={};logs={}
    for name,weights in [('initial',initial['actor']),('step8192',trained['actor'])]:
        actor=ACTSACActor(ACTConfig(**initial['model_config'])).eval();actor.load_state_dict(weights)
        with torch.no_grad():
            preds[name]=torch.cat([actor.get_eval_action(x) for x in obs.split(64)]).numpy()
            raws[name]=torch.cat([actor.action_head(actor.features(x)) for x in obs.split(64)]).numpy()
            logs[name]=torch.cat([actor(x)[1] for x in obs.split(64)]).numpy()
        states[name]=dict(saturated_per_action=(abs(raws[name])>=1-1e-6).mean(0).tolist(),
            raw_action_abs_max=np.abs(raws[name]).max(0).tolist(),mean_log_std=logs[name].mean(0).tolist())
    norms={n:float((v-initial['actor'][n]).norm()) for n,v in trained['actor'].items()}
    np.savez_compressed(out/'arrays.npz',observations=obs.numpy(),initial=preds['initial'],step8192=preds['step8192'],
        initial_raw=raws['initial'],step8192_raw=raws['step8192'],initial_log_std=logs['initial'],step8192_log_std=logs['step8192'])
    write(out/'result.json',dict(states=states,action_drift_mse_per_dim=((preds['step8192']-preds['initial'])**2).mean(0).tolist(),
        parameter_delta_l2=norms,source_train_episode_ids=split['train_episode_ids'][:20],used_states=len(obs),
        training_interactions=0,final_test_accessed=False,conclusion='Fixed-input observational drift only; no causal identification and no mechanism change'))

if __name__=='__main__':main()
