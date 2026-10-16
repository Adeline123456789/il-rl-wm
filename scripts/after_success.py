"""Run one explicit command after one successful prerequisite; never retry failures."""
import argparse
import datetime as dt
import json
import os
from pathlib import Path
import subprocess
import time


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--prerequisite', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--max-wait-seconds', type=float, default=900)
    p.add_argument('command', nargs=argparse.REMAINDER)
    args = p.parse_args()
    command = args.command[1:] if args.command[:1] == ['--'] else args.command
    assert command and args.max_wait_seconds > 0
    args.output.mkdir(exist_ok=False)
    record = dict(pid=os.getpid(), prerequisite=str(args.prerequisite), command=command, started_at=dt.datetime.now(dt.timezone.utc).isoformat(), phase='waiting', exit_code=None)
    def write():
        temp=args.output/'status.tmp';temp.write_text(json.dumps(record,indent=2));temp.replace(args.output/'status.json')
    write()
    deadline=time.monotonic()+args.max_wait_seconds
    try:
        while True:
            if args.prerequisite.exists():
                item=json.loads(args.prerequisite.read_text())
                if 'phase' in item:
                    assert item['phase'] != 'failed', item
                    ready=item['phase']=='completed'
                else:
                    ready=item.get('ended_at') is not None
                    if ready: assert item.get('exit_code')==0 and item.get('stop_reason') is None, item
                if ready: break
            assert time.monotonic()<deadline, 'Prerequisite wait limit exceeded'
            time.sleep(5)
        record['phase']='running';write()
        with (args.output/'command.log').open('w') as log:
            child=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT)
            record['child_pid']=child.pid;write()
            record['exit_code']=child.wait()
        assert record['exit_code']==0, 'Dependent command failed; no retries'
        record['phase']='completed'
    except BaseException as error:
        record['phase']='failed';record['error']=repr(error)
        raise
    finally:
        record['ended_at']=dt.datetime.now(dt.timezone.utc).isoformat();write()


if __name__ == '__main__': main()
