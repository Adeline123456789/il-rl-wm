"""CPU-only action/initial-state audit of frozen T0 and completed D1 traces."""
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from scripts.distill_act_sac import make_student
from scripts.train_act_teacher import write


def main():
    torch.set_num_threads(2)
    out=Path('artifacts/act-d1-diagnosis-002');out.mkdir(exist_ok=False)
    t=np.load('artifacts/act-t0-k30-seed0-001/final_validation_trace.npz')
    s=np.load('artifacts/act-d1-seed0-001/sac_mean/final_validation_trace.npz')
    actor=make_student(-2).eval()
    actor.load_state_dict(torch.load('releases/act-d1-negative-001/weights.pt',map_location='cpu',weights_only=False)['actor'])
    with torch.no_grad():
        x=torch.as_tensor(s['obs'].reshape(-1,42))
        actions=torch.cat([actor.get_eval_action(batch) for batch in x.split(256)]).numpy()
    action_error=float(abs(actions-s['action'].reshape(-1,4)).max())
    initial_error=float(abs(t['obs'][::200]-s['obs'][::200]).max())
    write(out/'result.json',dict(mean_action_cpu_recompute_max_abs=action_error,
        matched_reset_initial_observation_max_abs=initial_error,initial_states=40,
        initial_error_per_dimension=abs(t['obs'][::200]-s['obs'][::200]).reshape(-1,42).max(0).tolist(),
        threshold=1e-4,all_8000_student_actions_within_unit_range=bool((abs(s['action'])<=1).all()),
        reward_sum_teacher=float(t['reward'].sum()),reward_sum_student=float(s['reward'].sum()),
        new_environment_interactions=0,used_for_training=False,
        passed=action_error<1e-4 and initial_error<1e-4,
        conclusion='Numerical action and matched-reset audit; inspect individual errors before attributing failure.'))

if __name__=='__main__':main()
