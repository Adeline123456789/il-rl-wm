"""Recompute F1 component errors and tail-aware ranks from saved real branches."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from scripts.diagnose_ranking import ranking_metrics


def moments(values):
    return dict(count=values.size,mean=float(values.mean()),mae=float(np.abs(values).mean()),
                rmse=float(np.sqrt(np.square(values).mean()))) if values.size else None


def summarize(raw,decomposition,report):
    valid=raw['matching_error'].max(axis=(0,1,2))<=report['matching_tolerance']
    tail=report['unobserved_tail_interval'][1]
    tolerance=max(report['finite_return_tie_tolerance'],tail)
    result=dict(valid_state_indices=np.flatnonzero(valid).tolist(),tail_aware_tolerance=tolerance,
                new_environment_interactions=0,targets={})
    for mode,name in enumerate(['first_action_student','sequence_student']):
        actual=raw['returns'][mode].mean(0).T
        rows=ranking_metrics(actual[valid],raw['score'][valid],raw['selected'][valid],tolerance)
        eligible=sum(row['rankable_pairs'] for row in rows)
        correct=sum(row['rankable_pairs']*(row['pairwise_accuracy'] or 0) for row in rows)
        result['targets'][name]=dict(states=rows,rankable_pairs=eligible,
            weighted_pairwise_accuracy=correct/eligible if eligible else None,
            near_tie_pairs=sum(row['near_tie_pairs'] for row in rows),
            mean_regret=float(np.mean([row['regret'] for row in rows])) if rows else None)
    result['sequence_score_error_components']={}
    for source,selection in [('il',slice(0,1)),('sac',slice(1,9))]:
        result['sequence_score_error_components'][source]={
            key:moments(decomposition[key][:,selection,valid])
            for key in ['reward_error','dynamics_terminal_error','real_state_value_residual']}
    result['sequence_minus_first_return']={source:moments(
        (raw['returns'][1]-raw['returns'][0])[:,selection,valid])
        for source,selection in [('il',slice(0,1)),('sac',slice(1,9))]}
    # Infinite-return residual is in [finite residual - tail, finite residual].
    residual=decomposition['real_state_value_residual'][...,valid]
    result['real_state_value_tail_bounds']=dict(
        below_observed_suffix_lower_bound_fraction=float((residual<0).mean()) if residual.size else None,
        above_observed_suffix_tail_upper_bound_fraction=float((residual>tail).mean()) if residual.size else None,
        definition='Fractions across observed valid branches, not confidence bounds on expected value; residual includes gamma^2 and is relative to the finite real suffix.')
    result['limitations']='Descriptive decomposition, not causal attribution. Component MAEs need not add because errors cancel. Tail-aware pair ranks address only the unobserved nonnegative return tail, not statistical uncertainty. Repeated branches and candidates are not independent training runs.'
    return result


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--experiment',type=Path,required=True)
    args=parser.parse_args();folder=args.experiment
    report=json.loads((folder/'result.json').read_text())
    with np.load(folder/'branches.npz') as raw,np.load(folder/'score_decomposition.npz') as decomposition:
        result=summarize(raw,decomposition,report)
    (folder/'analysis.json').write_text(json.dumps(result,indent=2)+'\n')


if __name__=='__main__':main()
