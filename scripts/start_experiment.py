"""Detach a bounded GPU experiment and return its durable PID/log locations."""
import argparse
import datetime as dt
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--id', required=True)
    parser.add_argument('--seconds', type=float, required=True)
    parser.add_argument('--memory-mib', type=int, default=20480)
    parser.add_argument('command', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ['--'] else args.command
    if Path(args.id).name != args.id or not command or args.seconds <= 0:
        parser.error('Use a plain experiment ID, positive duration and command')
    directory = ROOT/'artifacts/launches'/args.id
    directory.mkdir(parents=True, exist_ok=False)
    launch = [sys.executable, str(ROOT/'scripts/run_experiment.py'), '--id', args.id,
              '--seconds', str(args.seconds), '--memory-mib', str(args.memory_mib), '--', *command]
    with (directory/'runner.log').open('w') as log:
        process = subprocess.Popen(launch, cwd=ROOT, stdin=subprocess.DEVNULL,
                                   stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    record = dict(experiment_id=args.id, pid=process.pid, started_at=dt.datetime.now(dt.timezone.utc).isoformat(),
                  command=launch, log=str(directory/'runner.log'))
    (directory/'launch.json').write_text(json.dumps(record, indent=2))
    print(json.dumps(record, indent=2))


if __name__ == '__main__':
    main()
