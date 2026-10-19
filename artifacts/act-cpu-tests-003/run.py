import datetime as dt, json, subprocess, time
from pathlib import Path
out=Path('artifacts/act-cpu-tests-003')
command=['.venv/bin/python', '-m', 'pytest', 'tests', '-q', '--junitxml=artifacts/act-cpu-tests-003/junit.xml']
start=time.monotonic()
record={'command':command,'started_at':dt.datetime.now(dt.timezone.utc).isoformat(),'cpu_only':True,'new_environment_interactions':0}
with (out/'stdout.log').open('w') as log:
    child=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT)
    record['child_pid']=child.pid
    (out/'status.json').write_text(json.dumps(record,indent=2)+'\n')
    record['exit_code']=child.wait()
record['elapsed_seconds']=time.monotonic()-start
record['ended_at']=dt.datetime.now(dt.timezone.utc).isoformat()
(out/'status.json').write_text(json.dumps(record,indent=2)+'\n')
raise SystemExit(record['exit_code'])
