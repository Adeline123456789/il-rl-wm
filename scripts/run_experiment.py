"""Single-job, bounded local runner; account GPU job wall time conservatively.

Usage: python scripts/run_experiment.py --id e0-smoke --seconds 300 -- CMD ...
Logs remain available after failures. A lost runner requires ledger reconciliation
before another GPU run; do not erase an unfinished manifest to bypass accounting.
"""
import argparse
import datetime as dt
import fcntl
import hashlib
import io
import json
import os
from pathlib import Path
import signal
import subprocess
import tarfile
import time

ROOT = Path(__file__).resolve().parents[1]


def stamp():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def write_json(path, value):
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')
    tmp.replace(path)


def gpu_sample():
    result = subprocess.run(
        ['nvidia-smi', '--query-gpu=index,memory.used,utilization.gpu',
         '--format=csv,noheader,nounits'], capture_output=True, text=True, timeout=10)
    if result.returncode:
        raise RuntimeError(result.stderr)
    rows = [[int(x.strip()) for x in row.split(',')] for row in result.stdout.splitlines()]
    if len(rows) != 1:
        raise RuntimeError('This batch requires exactly one visible physical GPU')
    return dict(zip(['index', 'memory_mib', 'utilization_percent'], rows[0]))


def host_sample(pid):
    # Child RSS/HWM excludes driver allocations; whole-device VRAM is sampled separately.
    status = {}
    try:
        for line in Path(f'/proc/{pid}/status').read_text().splitlines():
            if line.startswith(('VmRSS:', 'VmHWM:')):
                key, value, _ = line.split()
                status[key.rstrip(':') + '_kib'] = int(value)
    except FileNotFoundError:
        pass
    for line in Path('/proc/meminfo').read_text().splitlines():
        if line.startswith('MemAvailable:'):
            status['host_mem_available_kib'] = int(line.split()[1])
    filesystem = os.statvfs(ROOT)
    status['workspace_available_bytes'] = filesystem.f_bavail * filesystem.f_frsize
    return status


def ensure_gpu_idle():
    result = subprocess.run(['nvidia-smi', '--query-compute-apps=pid',
                             '--format=csv,noheader,nounits'], capture_output=True, text=True, timeout=10)
    if result.returncode or result.stdout.strip():
        raise RuntimeError('GPU compute process already exists or process query failed; defer this job')


def job_budget(state, previous, experiment, requested, authorization=None):
    """Separate explicitly authorized batches; never borrow the original balance."""
    if authorization is None:
        used = sum(m['elapsed_seconds'] for m in previous if not m.get('budget_id'))
        return used, min(requested, state['budget']['gpu_hours_limit'] * 3600 - used)
    assert authorization['status'] == 'authorized'
    budget_id = authorization['budget_id']
    jobs = authorization['jobs']
    assert experiment in jobs, 'Experiment is outside this authorization'
    job = jobs[experiment]
    assert requested <= job['cap_seconds']
    charged = [m for m in previous if m.get('budget_id') == budget_id]
    used = sum(m['elapsed_seconds'] for m in charged)
    category_used = sum(m['elapsed_seconds'] for m in charged
                        if jobs[m['experiment_id']]['category'] == job['category'])
    remaining = min(authorization['total_seconds'] - used,
                    authorization['category_caps'][job['category']] - category_used)
    # Keep space for the runner's existing bounded termination/cleanup.
    return used, min(requested, remaining - 12)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--id', required=True)
    parser.add_argument('--seconds', required=True, type=float)
    parser.add_argument('--memory-mib', type=int, default=20480)
    parser.add_argument('--budget-file', type=Path, help='Explicit separate batch authorization; default retains original 24-hour ledger')
    parser.add_argument('command', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ['--'] else args.command
    if not command or args.seconds <= 0 or Path(args.id).name != args.id:
        parser.error('Need a command, positive duration and plain experiment ID')
    (ROOT / '.cache').mkdir(exist_ok=True)
    with (ROOT / '.cache/gpu-job.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return run_locked(args, command)


def run_locked(args, command):
    manifests = list((ROOT / 'artifacts').glob('*/manifest.json'))
    previous = [json.loads(p.read_text()) for p in manifests]
    if any(m.get('ended_at') is None for m in previous):
        raise RuntimeError('Unfinished job manifest: reconcile elapsed budget first')
    state_path = ROOT / 'research_state.json'
    state = json.loads(state_path.read_text())
    authorization = json.loads(args.budget_file.read_text()) if args.budget_file else None
    if authorization:
        assert command == authorization['jobs'][args.id]['command'], 'Command differs from authorized registry'
    used, duration = job_budget(state, previous, args.id, args.seconds, authorization)
    if duration <= 0:
        raise RuntimeError('Batch budget exhausted')
    out = ROOT / 'artifacts' / args.id
    out.mkdir(exist_ok=False)
    manifest = {'experiment_id': args.id, 'started_at': stamp(), 'ended_at': None,
                'command': command, 'cwd': str(ROOT), 'limit_seconds': duration,
                'memory_limit_mib': args.memory_mib, 'runner_pid': os.getpid(),
                'code_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
                'code_dirty': bool(subprocess.check_output(['git', 'status', '--porcelain'], cwd=ROOT)),
                'exit_code': None, 'stop_reason': None}
    if authorization:
        manifest.update(budget_id=authorization['budget_id'], budget_file=str(args.budget_file),
                        budget_authorization_sha256=hashlib.sha256(args.budget_file.read_bytes()).hexdigest())
    patch = subprocess.check_output(['git', 'diff', 'HEAD'], cwd=ROOT)
    (out / 'working-tree.patch').write_bytes(patch)
    manifest['tracked_patch_sha256'] = hashlib.sha256(patch).hexdigest()
    write_json(out / 'manifest.json', manifest)
    sources = {}
    snapshot = out / 'source_snapshot.tar.gz'
    with tarfile.open(snapshot, 'w:gz') as archive:
        for directory in ['scripts', 'configs', 'tests', 'src', 'vendor', 'external']:
            for path in sorted((ROOT / directory).rglob('*')):
                if path.is_file() and '__pycache__' not in path.parts:
                    name = str(path.relative_to(ROOT)); data = path.read_bytes()
                    sources[name] = hashlib.sha256(data).hexdigest()
                    item = tarfile.TarInfo(name); item.size = len(data)
                    archive.addfile(item, io.BytesIO(data))
    write_json(out / 'source_hashes.json', sources)
    manifest['source_snapshot_sha256'] = hashlib.sha256(snapshot.read_bytes()).hexdigest()
    write_json(out / 'manifest.json', manifest)
    process = None
    started = time.monotonic()
    try:
        ensure_gpu_idle()
        initial = gpu_sample()
        if initial['memory_mib'] >= args.memory_mib:
            raise RuntimeError('GPU already above memory target')
        with (out / 'stdout.log').open('w') as stdout, (out / 'stderr.log').open('w') as stderr, (out / 'resources.jsonl').open('w') as resources:
            env = dict(os.environ, CUDA_VISIBLE_DEVICES='0', PYTHONUNBUFFERED='1')
            process = subprocess.Popen(command, cwd=ROOT, stdout=stdout, stderr=stderr,
                                       env=env, start_new_session=True)
            manifest['child_pid'] = process.pid
            write_json(out / 'manifest.json', manifest)
            while process.poll() is None:
                sample = dict(timestamp=stamp(), elapsed_seconds=time.monotonic()-started, **gpu_sample(), **host_sample(process.pid))
                resources.write(json.dumps(sample)+'\n'); resources.flush()
                if sample['memory_mib'] >= args.memory_mib:
                    manifest['stop_reason'] = 'memory_target'
                    break
                if sample['elapsed_seconds'] >= duration:
                    manifest['stop_reason'] = 'time_budget'
                    break
                time.sleep(min(2, max(0.01, duration-sample['elapsed_seconds'])))
    except BaseException as exc:
        manifest['stop_reason'] = f'{type(exc).__name__}: {exc}'
        raise
    finally:
        if process is not None:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
            manifest['exit_code'] = process.returncode
        manifest['elapsed_seconds'] = time.monotonic()-started
        manifest['ended_at'] = stamp()
        write_json(out / 'manifest.json', manifest)
        state = json.loads(state_path.read_text())
        if authorization:
            state.setdefault('additional_budgets', {})[authorization['budget_id']] = dict(
                authorization=str(args.budget_file), limit_seconds=authorization['total_seconds'],
                gpu_job_seconds_used=used + manifest['elapsed_seconds'])
        else:
            state['budget']['gpu_job_seconds_used'] = used + manifest['elapsed_seconds']
        state['experiments'].append(args.id)
        write_json(state_path, state)
    print(json.dumps(manifest, indent=2))
    return 0 if manifest['exit_code'] == 0 and manifest['stop_reason'] is None else 1


if __name__ == '__main__':
    def stop(signum, frame):
        raise KeyboardInterrupt(f'Received signal {signum}; terminating child and saving ledger')
    signal.signal(signal.SIGTERM, stop)
    raise SystemExit(main())
