import importlib.util
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import tarfile
import unittest
from unittest.mock import patch

SPEC = importlib.util.spec_from_file_location('runner', Path(__file__).parents[1]/'scripts/run_experiment.py')
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)


class RunnerTests(unittest.TestCase):
    def test_failure_and_timeout_are_recorded_and_budgeted(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root/'artifacts').mkdir()
            (root/'scripts').mkdir()
            untracked = root/'scripts/untracked.py'
            untracked.write_bytes(b'print("untracked launch source")\n')
            (root/'research_state.json').write_text(json.dumps({'budget': {'gpu_hours_limit': 1}, 'experiments': []}))
            sample = {'index': 0, 'memory_mib': 100, 'utilization_percent': 0}
            with patch.object(runner, 'ROOT', root), patch.object(runner, 'gpu_sample', return_value=sample), patch.object(runner, 'ensure_gpu_idle'), patch.object(runner.subprocess, 'check_output', side_effect=lambda cmd, **kw: 'test-commit\n' if kw.get('text') else b''):
                for name, duration, code in [('failure', '5', 'raise SystemExit(7)'), ('timeout', '0.05', 'import time; time.sleep(20)')]:
                    with patch.object(sys, 'argv', ['runner', '--id', name, '--seconds', duration, '--', sys.executable, '-c', code]):
                        self.assertEqual(runner.main(), 1)
                failed = json.loads((root/'artifacts/failure/manifest.json').read_text())
                timed = json.loads((root/'artifacts/timeout/manifest.json').read_text())
                self.assertEqual(failed['exit_code'], 7)
                self.assertEqual(timed['stop_reason'], 'time_budget')
                self.assertIsNotNone(timed['ended_at'])
                state = json.loads((root/'research_state.json').read_text())
                self.assertAlmostEqual(state['budget']['gpu_job_seconds_used'], failed['elapsed_seconds']+timed['elapsed_seconds'])
                snapshot = root/'artifacts/failure/source_snapshot.tar.gz'
                self.assertEqual(hashlib.sha256(snapshot.read_bytes()).hexdigest(), failed['source_snapshot_sha256'])
                hashes = json.loads((snapshot.parent/'source_hashes.json').read_text())
                with tarfile.open(snapshot) as archive:
                    captured = archive.extractfile('scripts/untracked.py').read()
                self.assertEqual(captured, untracked.read_bytes())
                self.assertEqual(hashlib.sha256(captured).hexdigest(), hashes['scripts/untracked.py'])

    def test_unfinished_run_blocks_new_work(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root/'artifacts/lost').mkdir(parents=True)
            (root/'artifacts/lost/manifest.json').write_text('{"ended_at":null}')
            with patch.object(runner, 'ROOT', root), patch.object(sys, 'argv', ['runner', '--id', 'new', '--seconds', '1', '--', 'true']):
                with self.assertRaisesRegex(RuntimeError, 'Unfinished'):
                    runner.main()


if __name__ == '__main__':
    unittest.main()
