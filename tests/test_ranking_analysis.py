import numpy as np
from scripts.summarize_ranking import summarize


def test_excludes_unmatched_state_and_separates_tail_ties():
    returns=np.zeros((2,2,9,2))
    returns[:,:,0,0]=1.5  # Within the 400-step unobserved-tail bound.
    returns[:,:,1,0]=5.
    returns[:,:,:,1]=1000.  # Excluded state must not contaminate statistics.
    matching=np.zeros_like(returns);matching[0,0,0,1]=.1
    score=np.tile(np.arange(9)[::-1],(2,1))
    raw=dict(returns=returns,matching_error=matching,score=score,selected=np.zeros(2,dtype=int))
    error=np.zeros((2,9,2));error[:,:,0]=-2.;error[:,:,1]=1000.
    report=dict(matching_tolerance=1e-5,unobserved_tail_interval=[0,1.8],finite_return_tie_tolerance=1.)
    result=summarize(raw,{key:error for key in ['reward_error','dynamics_terminal_error','real_state_value_residual']},report)
    assert result['valid_state_indices']==[0]
    assert result['targets']['sequence_student']['rankable_pairs']==8
    assert result['targets']['sequence_student']['mean_regret']==3.5
    assert result['sequence_score_error_components']['sac']['reward_error']['mean']==-2.
    assert result['real_state_value_tail_bounds']['below_observed_suffix_lower_bound_fraction']==1.
