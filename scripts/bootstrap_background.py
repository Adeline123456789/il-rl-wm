"""Bounded background environment setup, followed by the first real smoke run.

Waits for the already-started downloads; never starts a second installer in the
same environment. Failures stop the chain for inspection by the research agent.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
STATUS = ROOT/'artifacts/bootstrap/background-status.json'


def status(phase, **extra):
    state = dict(pid=os.getpid(), phase=phase, updated_unix=time.time(), **extra)
    temp = STATUS.with_suffix('.tmp')
    temp.write_text(json.dumps(state, indent=2))
    temp.replace(STATUS)
    print(json.dumps(state), flush=True)


def alive(pid):
    stat = Path(f'/proc/{pid}/stat')
    return stat.exists() and stat.read_text().split(') ')[1][0] != 'Z'


def run(command, log, timeout=3600):
    status('running', command=command, log=log)
    with (ROOT/log).open('w') as stream:
        result = subprocess.run(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT, timeout=timeout)
    if result.returncode:
        raise RuntimeError(f'exit {result.returncode}: {command}; see {log}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--wait-pids', nargs='+', type=int, required=True)
    parser.add_argument('--require-local-runtime', action='store_true')
    args = parser.parse_args()
    deadline = time.monotonic()+7200
    status('waiting_existing_downloads', waiting_pids=args.wait_pids, timeout_seconds=7200)
    while any(alive(pid) for pid in args.wait_pids):
        if time.monotonic() > deadline:
            raise TimeoutError('Existing downloads exceeded 2h setup wait; inspect before retry')
        time.sleep(10)
    if args.require_local_runtime and not (ROOT/'artifacts/bootstrap/local-runtime-manifest.json').exists():
        raise RuntimeError('Local runtime copy did not finish successfully; inspect copy-runtime.log')
    python = str(ROOT/'.venv/bin/python')
    run([python, '-c', 'import importlib.metadata as m; print(m.version("torch"))'], 'artifacts/bootstrap/torch-metadata.log', 60)
    commit = subprocess.check_output(['git','-C','.cache/upstreams/ManiSkill','rev-parse','HEAD'], cwd=ROOT, text=True).strip()
    if commit != '62ff3a5896b4d5b4cf0ac4c8d79afe600c9404a3' or not (ROOT/'.cache/upstreams/ManiSkill/setup.py').exists():
        raise RuntimeError('Upstream checkout missing or wrong commit')
    run(['uv','pip','install','--python',python,'--index-url','https://pypi.org/simple',
         '-r','configs/environment.in','-e','.cache/upstreams/ManiSkill'], 'artifacts/bootstrap/install-maniskill.log')
    run(['uv','pip','check','--python',python], 'artifacts/bootstrap/dependency-check.log', 60)
    run([python, '-c', 'import torch; print(torch.__version__)'], 'artifacts/bootstrap/torch-import.log', 60)
    freeze = subprocess.check_output(['uv','pip','freeze','--python',python], cwd=ROOT, text=True)
    (ROOT/'requirements.lock').write_text(freeze.replace(str(ROOT), '.'))
    run([python,'scripts/run_experiment.py','--id','e0-pickcube-smoke-001','--seconds','300','--',
         python,'scripts/smoke_env.py','--env-id','PickCube-v1','--output','artifacts/e0-pickcube-smoke-001'],
        'artifacts/bootstrap/e0-launch.log', 330)
    status('completed_first_smoke', next_action='Inspect E0 evidence, then remaining task smoke tests and official demos')


if __name__ == '__main__':
    try:
        main()
    except BaseException as exc:
        status('failed', error=str(exc))
        traceback.print_exc()
        raise SystemExit(1)
