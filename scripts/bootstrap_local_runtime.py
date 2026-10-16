"""Copy only installed PyTorch runtime distributions into an isolated venv.

Source files are read-only; verify installed RECORD hashes while copying.
Do not copy unrelated frameworks, editable packages, or paths outside purelib.
The public package versions remain independently installable from PyPI.
"""
import base64
import argparse
import hashlib
from importlib.metadata import distributions
import json
from pathlib import Path
import shutil

SOURCE = Path('/home/midea/miniconda3/envs/avp_clean/lib/python3.10/site-packages')
TARGET = Path('.venv/lib/python3.10/site-packages')


def main(SOURCE=SOURCE,TARGET=TARGET,manifest_path=Path('artifacts/bootstrap/local-runtime-manifest.json')):
    assert TARGET.exists()
    manifest = dict(source=str(SOURCE), target=str(TARGET), packages=[], skipped_entrypoints=[])
    names = {'torch', 'torchvision', 'triton'}
    for dist in distributions(path=[str(SOURCE)]):
        name = dist.metadata['Name']
        if name.lower() not in names and not name.lower().startswith('nvidia-'):
            continue
        entry = dict(name=name, version=dist.version, files=0, bytes=0, verified_hashes=0)
        for relative in dist.files:
            if '..' in relative.parts or relative.is_absolute():
                manifest['skipped_entrypoints'].append(str(relative))
                continue
            src, dst = SOURCE/relative, TARGET/relative
            if not src.is_file():
                raise FileNotFoundError(src)
            if relative.hash:
                digest = hashlib.new(relative.hash.mode)
                with src.open('rb') as stream:
                    for block in iter(lambda: stream.read(2**20), b''):
                        digest.update(block)
                encoded = base64.urlsafe_b64encode(digest.digest()).rstrip(b'=').decode()
                if encoded != relative.hash.value:
                    raise RuntimeError(f'Installed RECORD mismatch: {src}')
                entry['verified_hashes'] += 1
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
            entry['files'] += 1
            entry['bytes'] += src.stat().st_size
        manifest['packages'].append(entry)
        print(json.dumps(entry), flush=True)
    found = {x['name'].lower() for x in manifest['packages']}
    assert names.issubset(found)
    manifest_path.write_text(json.dumps(manifest, indent=2))


if __name__ == '__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--source',type=Path,default=SOURCE);parser.add_argument('--target',type=Path,default=TARGET)
    parser.add_argument('--manifest',type=Path,default=Path('artifacts/bootstrap/local-runtime-manifest.json'))
    args=parser.parse_args();main(args.source,args.target,args.manifest)
