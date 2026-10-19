import hashlib,json,tarfile,sys
from pathlib import Path
out=Path('artifacts/act-t3-source-audit-001');folder=Path('artifacts/act-t3-online-seed0-001')
try:
    manifest=json.loads((folder/'manifest.json').read_text())
    assert hashlib.sha256((folder/'working-tree.patch').read_bytes()).hexdigest()==manifest['tracked_patch_sha256']
    assert hashlib.sha256((folder/'source_snapshot.tar.gz').read_bytes()).hexdigest()==manifest['source_snapshot_sha256']
    hashes=json.loads((folder/'source_hashes.json').read_text())
    with tarfile.open(folder/'source_snapshot.tar.gz') as archive:
        members=archive.getmembers();assert set(hashes)=={m.name for m in members}
        for member in members:
            assert hashlib.sha256(archive.extractfile(member).read()).hexdigest()==hashes[member.name]
    result=dict(source_files_verified=len(hashes),actual_code_commit=manifest['code_commit'],dirty_source_captured=True,raw_patch_preserved=True,new_environment_interactions=0,exit_code=0)
except BaseException as error:
    result=dict(exit_code=1,error=repr(error))
(out/'result.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result));sys.exit(result['exit_code'])
