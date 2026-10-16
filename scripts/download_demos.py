"""Pin dataset revision, then use ManiSkill's official downloader unchanged."""
import argparse
import hashlib
import json
from pathlib import Path
import time

from mani_skill.utils import download_demo


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env-id', required=True)
    parser.add_argument('--source', default='artifacts/bootstrap/demo-source.json')
    parser.add_argument('--output', default='data/official')
    args = parser.parse_args()
    source = json.loads(Path(args.source).read_text())
    assert source['required_files'][f'demos/{args.env_id}.zip']
    revision = source['revision']
    url = f'https://huggingface.co/datasets/{source["repository"]}/resolve/{revision}/demos/{args.env_id}.zip?download=true'
    target = Path(args.output)/args.env_id
    if target.exists():
        raise FileExistsError(f'{target}: inspect existing data before re-downloading')
    # The upstream tool exposes a mutable source table; pin only this URL.
    download_demo.DATASET_SOURCES[args.env_id].raw_dataset_url = url
    started = time.monotonic()
    download_demo.main(download_demo.parse_args([args.env_id, '-o', args.output]))
    manifest = dict(env_id=args.env_id, source_repository=source['repository'], revision=revision,
                    url=url, elapsed_seconds=time.monotonic()-started, files={})
    for path in sorted(target.rglob('*')):
        if path.is_file():
            manifest['files'][str(path)] = dict(bytes=path.stat().st_size, sha256=sha256(path))
    if not manifest['files']:
        raise RuntimeError('Official downloader completed but no expected files were found')
    destination = Path('artifacts/datasets'); destination.mkdir(parents=True, exist_ok=True)
    (destination/f'{args.env_id}-raw-manifest.json').write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2))


if __name__ == '__main__':
    main()
