"""Recompute experiment status/resources/episode aggregates from raw artifact files."""
import argparse
import json
from pathlib import Path


def rows(path):
    return [json.loads(s) for s in path.read_text().splitlines() if s.strip()] if path.exists() else []


def summarize(root):
    experiments=[]
    for path in sorted(root.glob('*/manifest.json')):
        m=json.loads(path.read_text());folder=path.parent
        resource=rows(folder/'resources.jsonl')
        item=dict(experiment_id=m['experiment_id'],commit=m['code_commit'],exit_code=m['exit_code'],stop_reason=m['stop_reason'],elapsed_seconds=m.get('elapsed_seconds'),completed=m['ended_at'] is not None,invalidated=(folder/'INVALIDATED.json').exists(),gpu_memory_mib_peak_sampled=max((r['memory_mib'] for r in resource),default=None),child_rss_kib_peak_sampled=max((r.get('VmRSS_kib',0) for r in resource),default=0) or None,resource_sampling_note='Sampled peaks can miss sub-2-second spikes; child RSS excludes other host processes and driver VRAM.')
        for name in ['result','interactions','model']:
            p=folder/f'{name}.json'
            if p.exists(): item[name]=json.loads(p.read_text())
        ep=rows(folder/'validation_episodes.jsonl');groups={}
        for row in ep:
            key=str(row.get('step',row.get('iteration')))
            groups.setdefault(row.get('method','policy'),{}).setdefault(key,[]).append(row)
        item['validation_by_method_and_checkpoint']={method:{k:dict(episodes=len(v),success_once=sum(x['success_once'] for x in v)/len(v),success_at_end=sum(x['success_at_end'] for x in v)/len(v),mean_return=sum(x['return'] for x in v)/len(v),completed_episode_interactions=sum(x['episode_len'] for x in v)) for k,v in checkpoints.items()} for method,checkpoints in groups.items()}
        if not item['completed']:
            item['limitation']='Running experiment: final counters and job duration are not yet available.'
        elif m['exit_code']!=0 and not (folder/'interactions.json').exists():
            item['limitation']='Interrupted/failed before live interaction counters were introduced; completed episode counts are lower bounds, not exact total steps.'
        experiments.append(item)
    return dict(experiments=experiments,completed_gpu_job_seconds=sum(x['elapsed_seconds'] or 0 for x in experiments),ongoing_experiments=[x['experiment_id'] for x in experiments if not x['completed']],definition='Conservative whole GPU-job walltime includes downloads, replay, validation and initialization. Failed and invalidated jobs are charged. No claims of convergence or statistical significance.')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=Path('artifacts'));p.add_argument('--output',type=Path,default=Path('artifacts/evidence-summary.json'));args=p.parse_args();args.output.write_text(json.dumps(summarize(args.root),indent=2))
