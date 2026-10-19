"""Archive completed historical run evidence, then verify every byte and recomputation."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import sys
import tarfile
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.summarize_historical_validation import summarize


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    plan = json.loads(Path('configs/final_evaluation_plan.json').read_text())
    expected = summarize(plan, Path('artifacts'))
    names = set(expected['runs'])
    pending = list(names)
    while pending:
        name = pending.pop()
        config = Path('artifacts') / name / 'sac_config.json'
        if not config.exists():
            continue
        values = json.loads(config.read_text())
        for key in ['bc_checkpoint', 'resume_checkpoint', 'wm_checkpoint', 'task_value_checkpoint']:
            if values.get(key):
                dependency = Path(values[key]).parent.name
                if dependency not in names:
                    names.add(dependency); pending.append(dependency)
    files = []
    for name in sorted(names):
        folder = Path('artifacts') / name
        manifest = json.loads((folder / 'manifest.json').read_text())
        assert manifest['ended_at'] and manifest['exit_code'] == 0 and manifest['stop_reason'] is None
        files.extend(p for p in sorted(folder.iterdir()) if p.is_file())
    assert all(p.suffix not in {'.pt', '.mp4', '.h5', '.hdf5'} for p in files)
    args.output.mkdir(parents=True, exist_ok=False)
    archive = args.output / 'raw-evidence.tar.gz'
    inventory = []
    with tarfile.open(archive, 'w:gz', compresslevel=6) as tar:
        for p in files:
            data = p.read_bytes()
            digest = hashlib.sha256(data).hexdigest()
            item = tarfile.TarInfo(str(p)); item.size = len(data); item.mtime = 0
            tar.addfile(item, io.BytesIO(data))
            assert hashlib.sha256(p.read_bytes()).hexdigest() == digest, 'Source changed during capture'
            inventory.append(dict(path=str(p), bytes=len(data), sha256=digest))
    # Verify the distributed archive itself, not only the source directory.
    with tempfile.TemporaryDirectory(prefix='historical-evidence-', dir='.cache') as temporary:
        root = Path(temporary)
        with tarfile.open(archive, 'r:gz') as tar:
            members = tar.getmembers()
            assert [m.name for m in members] == [r['path'] for r in inventory]
            for member, row in zip(members, inventory):
                assert member.isfile() and not Path(member.name).is_absolute() and '..' not in Path(member.name).parts
                data = tar.extractfile(member).read()
                assert len(data) == row['bytes'] and hashlib.sha256(data).hexdigest() == row['sha256']
                target = root / member.name; target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(data)
        assert summarize(plan, root / 'artifacts') == expected
    chunks = []
    with archive.open('rb') as stream:
        for index in range((archive.stat().st_size + 4194303) // 4194304):
            data = stream.read(4194304)
            part = args.output / f'raw-evidence.tar.gz.part{index:03d}'
            part.write_bytes(data)
            chunks.append(dict(file=part.name, bytes=len(data), sha256=hashlib.sha256(data).hexdigest()))
    record = dict(files=inventory, experiments=sorted(names), chunks=chunks,
        source_bytes=sum(r['bytes'] for r in inventory), archive_bytes=archive.stat().st_size,
        archive_sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),
        extracted_recomputation_equal=True, conditions=len(expected['conditions']),
        excludes='Large checkpoints, replay, original demonstrations, videos, active ACT online runs; no final-test data.')
    (args.output / 'inventory.json').write_text(json.dumps(record, indent=2) + '\n')
    print(json.dumps({k:v for k,v in record.items() if k not in ['files','experiments']}))


if __name__ == '__main__':
    main()
