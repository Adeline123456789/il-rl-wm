import json,h5py,numpy as np
from pathlib import Path
cpu=Path('data/replay/e0-peg-cpu-replay-smoke-002/trajectory.state.pd_ee_delta_pose.physx_cpu.h5')
gpu=Path('data/replay/e0-peg-gpu-replay-smoke-002/trajectory.state.pd_ee_delta_pose.physx_cuda.h5')
cm=json.loads(cpu.with_suffix('.json').read_text());gm=json.loads(gpu.with_suffix('.json').read_text())
rows=[]
with h5py.File(cpu) as cf,h5py.File(gpu) as gf:
 for e in gm['episodes']:
  a=cf[f"traj_{e['source_episode_id']}"];b=gf[f"traj_{e['episode_id']}"]
  rows.append(dict(original_episode_id=e['original_episode_id'],source_episode_id=e['source_episode_id'],target_episode_id=e['episode_id'],cpu_success=bool(a['success'][-1]),gpu_success=bool(b['success'][-1]),initial_obs_max_abs_error=float(np.max(np.abs(a['obs'][0]-b['obs'][0]))),initial_obs_abs_error=np.abs(a['obs'][0]-b['obs'][0]).tolist()))
Path('artifacts/peg-replay-transfer-diagnosis-001/result.json').write_text(json.dumps(dict(episodes=len(rows),cpu_success=sum(r['cpu_success'] for r in rows),gpu_success=sum(r['gpu_success'] for r in rows),max_initial_obs_error=max(r['initial_obs_max_abs_error'] for r in rows),episodes_detail=rows),indent=2))
print(dict(episodes=len(rows),cpu_success=sum(r['cpu_success'] for r in rows),gpu_success=sum(r['gpu_success'] for r in rows),max_initial_obs_error=max(r['initial_obs_max_abs_error'] for r in rows)))
