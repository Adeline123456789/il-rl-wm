"""E5 real-snapshot candidate expansion and measured GPU latency, no env.step."""
import argparse
import json
from pathlib import Path
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from scripts.load_components import sac_actor, bc_actor, world_model
from scripts.planning import generate_candidates


def main():
    p=argparse.ArgumentParser();p.add_argument('--sac',type=Path,required=True);p.add_argument('--bc',type=Path,required=True);p.add_argument('--wm',type=Path,required=True);p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    torch.set_num_threads(4);torch.manual_seed(2026);args.output.mkdir(exist_ok=True,parents=True)
    snapshot=torch.load(args.sac,map_location='cpu');actor=sac_actor(snapshot,'cuda').eval();actor.requires_grad_(False)
    model=world_model(torch.load(args.wm,map_location='cpu'),'cuda').eval();model.requires_grad_(False)
    obs_all=snapshot['replay']['obs'].flatten(0,1);indices=torch.randperm(len(obs_all))[:32]
    obs=obs_all[indices].cuda();il=bc_actor(args.bc,obs.shape[-1],actor.fc_mean.out_features,'cuda')
    summaries={}
    for name,teacher in [('with_il',il),('without_il',None)]:
        torch.manual_seed(1234);first=generate_candidates(obs,actor,model,teacher)
        torch.manual_seed(1234);repeat=generate_candidates(obs,actor,model,teacher)
        assert torch.equal(first['actions'],repeat['actions'])
        assert first['actions'].shape[:3]==(32,9,2)
        assert first['states'].shape==(32,9,3,obs.shape[-1])
        assert torch.equal(first['states'][:,:,0],obs[:,None,:].expand(32,9,-1))
        assert first['actions'].abs().max()<=1.00001 and torch.isfinite(first['states']).all()
        if teacher is not None:torch.testing.assert_close(first['actions'][:,0,1],il(first['states'][:,0,1]))
        start_sac=1 if teacher is not None else 0
        diversity=first['actions'][:,start_sac:,0].std(dim=1).mean()
        assert diversity>0
        for _ in range(10):generate_candidates(obs,actor,model,teacher)
        torch.cuda.synchronize();start=time.perf_counter()
        for _ in range(100):generate_candidates(obs,actor,model,teacher)
        torch.cuda.synchronize();latency=(time.perf_counter()-start)*10
        arrays={k:v.cpu().numpy() for k,v in first.items() if isinstance(v,torch.Tensor)}
        np.savez_compressed(args.output/(name+'.npz'),sample_indices=indices.numpy(),**arrays)
        summaries[name]=dict(sources=first['sources'],latency_ms_per_batch32=latency,mean_sac_candidate_std=float(diversity),
                            predicted_state_change_mean=float((first['states'][:,:,1:]-first['states'][:,:,:-1]).abs().mean()))
    (args.output/'result.json').write_text(json.dumps(dict(sac=str(args.sac),bc=str(args.bc),wm=str(args.wm),new_environment_interactions=0,
        measured_generation_only=True,results=summaries,torch_peak_allocated_bytes=torch.cuda.max_memory_allocated()),indent=2))


if __name__=='__main__':main()
