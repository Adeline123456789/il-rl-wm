"""Bounded, detached setup of the external baseline; never changes main .venv."""
import json
import os
from pathlib import Path
import subprocess
import time
import traceback
import urllib.request

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'artifacts/bootstrap/rlpd-setup'
COMMIT='7bd660f05dbda1ec8881bcfb250a8d1881532dbb'


def main():
    OUT.mkdir(exist_ok=False);start=time.monotonic()
    def status(phase,**extra):(OUT/'status.json').write_text(json.dumps(dict(phase=phase,elapsed_seconds=time.monotonic()-start,**extra),indent=2))
    def run(command,cwd=ROOT,seconds=1800):
        with (OUT/'commands.jsonl').open('a') as f:f.write(json.dumps(dict(command=command,cwd=str(cwd)))+'\n')
        subprocess.run(command,cwd=cwd,check=True,timeout=seconds,env=os.environ|{'JAX_PLATFORMS':'cpu','XLA_PYTHON_CLIENT_PREALLOCATE':'false'})
    try:
        status('rfcl_checkout')
        run(['git','-c','http.version=HTTP/1.1','clone','--depth','1','https://github.com/StoneT2000/rfcl','.cache/upstreams/rfcl'])
        run(['git','checkout','--detach',COMMIT],ROOT/'.cache/upstreams/rfcl')
        assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT/'.cache/upstreams/rfcl',text=True).strip()==COMMIT
        status('isolated_python_runtime')
        assert not (ROOT/'.venv-rlpd').exists()
        run(['uv','venv','--python',str(ROOT/'.venv/bin/python'),'.venv-rlpd'])
        run(['python','scripts/bootstrap_local_runtime.py','--source','.venv/lib/python3.10/site-packages','--target','.venv-rlpd/lib/python3.10/site-packages','--manifest',str(OUT/'runtime-copy.json')])
        status('install_jax_dependencies')
        run(['uv','pip','install','--python',str(ROOT/'.venv-rlpd/bin/python'),'-r','requirements.in','--find-links','https://storage.googleapis.com/jax-releases/jax_cuda_releases.html'],ROOT/'external/rlpd',seconds=3600)
        status('cpu_import_check')
        run(['uv','pip','check','--python',str(ROOT/'.venv-rlpd/bin/python')])
        run([str(ROOT/'.venv-rlpd/bin/python'),'-c','import jax,torch,mani_skill,rfcl; from rfcl.agents.sac import SAC,ActorCritic,SACConfig; print(dict(jax=jax.__version__,torch=torch.__version__,devices=str(jax.devices())))'])
        lock=subprocess.check_output(['uv','pip','freeze','--python',str(ROOT/'.venv-rlpd/bin/python')],cwd=ROOT,text=True)
        (ROOT/'external/rlpd/requirements.lock').write_text(lock)
        status('completed',exit_code=0)
    except BaseException:
        status('failed',exit_code=1,error=traceback.format_exc());raise


if __name__=='__main__':main()
