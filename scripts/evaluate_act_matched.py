"""Fixed-checkpoint ACT/D1 evaluation with matched batch resets; no training."""
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from scripts.act_teacher import ACTConfig,ACTTeacher
from scripts.distill_act_sac import make_student
from scripts.train_act_teacher import closed_loop,write
from scripts.interaction_counter import InteractionCounter


def main():
    torch.set_num_threads(2)
    out=Path('artifacts/act-matched-eval-001');counter=InteractionCounter(out/'interactions.json')
    models=[]
    for k,name in [(8,'act-t0-seed0-001'),(16,'act-t0-k16-seed0-001'),(30,'act-t0-k30-seed0-001')]:
        full=torch.load(Path('checkpoints')/name/'final.pt',map_location='cpu',weights_only=False)
        teacher=ACTTeacher(42,4,ACTConfig(**full['model_config'])).to('cuda').eval()
        teacher.load_state_dict(full['model']);teacher.requires_grad_(False)
        models += [(f'act_k{k}_first',teacher,False),(f'act_k{k}_ensemble',teacher,True)]
    student=make_student(-2).to('cuda').eval()
    student.load_state_dict(torch.load('releases/act-d1-negative-001/weights.pt',map_location='cpu',weights_only=False)['actor'])
    models += [('student_mean',student.get_eval_action,False),('student_stochastic',lambda x:student.get_action(x)[0],False)]
    results={};initial=None
    for name,model,temporal in models:
        directory=out/name;directory.mkdir();torch.manual_seed(10000)
        result=closed_loop(model,directory,10000,40,counter,video=True,temporal=temporal,method=name,matched_resets=True)
        trace=np.load(directory/('final_ensemble_trace.npz' if temporal else 'final_validation_trace.npz'))
        current=trace['obs'][::200]
        if initial is None:initial=current.copy()
        result['initial_obs_max_abs_from_first_condition']=float(abs(current-initial).max())
        assert result['initial_obs_max_abs_from_first_condition']<1e-4
        results[name]=result
        write(out/'partial_results.json',results)
    write(out/'result.json',dict(methods=results,protocol='matched batch reset seeds 10000,10008,10016,10024,10032; state42/pd_ee_delta_pos/sparse/GPU/horizon200',
        training_interactions=0,checkpoint_selection='original fixed-final K30 frozen before D1; no reselection from this corrected evaluation',
        final_test_accessed=False,torch_peak_allocated_bytes=torch.cuda.max_memory_allocated()))

if __name__=='__main__':main()
