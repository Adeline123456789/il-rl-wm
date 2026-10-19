"""Execute the frozen T2--T6 budgeted comparison, one GPU job at a time."""
import argparse,hashlib,json,os,subprocess,sys,time
from pathlib import Path


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--plan',default='artifacts/act-online-budget-001/plan.json')
    parser.add_argument('--output',default='artifacts/act-online-batch-001')
    args=parser.parse_args()
    plan_path=Path(args.plan)
    plan=json.loads(plan_path.read_text());assert plan['fits']
    assert json.loads(Path('artifacts/act-online-smokes-001/status.json').read_text())['phase']=='completed'
    out=Path(args.output);out.mkdir(exist_ok=False)
    state=dict(pid=os.getpid(),phase='running',completed=[],current=None,started_at=time.time(),
        plan_sha256=hashlib.sha256(plan_path.read_bytes()).hexdigest(),new_budget_granted=False)
    def save():
        p=out/'status.tmp';p.write_text(json.dumps(state,indent=2)+'\n');p.replace(out/'status.json')
    save()
    try:
        for job in plan['jobs']:
            budget=json.loads(Path('research_state.json').read_text())['budget']
            assert budget['gpu_hours_limit']*3600-budget['gpu_job_seconds_used']-budget['final_evaluation_reserve_seconds']>=job['seconds']
            state['current']=job['id'];save()
            subprocess.run([sys.executable,'scripts/run_experiment.py','--id',job['id'],'--seconds',str(job['seconds']),'--',*job['command']],check=True)
            actual=json.loads((Path('artifacts')/job['id']/'actor_initialization.json').read_text())
            assert [actual['critic1_hash'],actual['critic2_hash']]==plan['fixed_critic_hashes']
            state['completed'].append(job['id']);save()
        state.update(phase='completed',exit_code=0,current=None)
    except BaseException as e:
        state.update(phase='failed',exit_code=1,error=repr(e));raise
    finally:
        state['ended_at']=time.time();save()

if __name__=='__main__':main()
