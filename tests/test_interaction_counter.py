import atexit
import json
from pathlib import Path
import signal
import tempfile
import unittest
import gymnasium as gym
from scripts.interaction_counter import InteractionCounter, CountedEnv


class FakeEnv(gym.Env):
    num_envs=3
    def __init__(self): self.fail=False
    def step(self, action):
        if self.fail: raise RuntimeError('Interrupted simulated step')
        return 0,0,False,False,{}


class CounterTests(unittest.TestCase):
    def test_failed_step_stays_attempted_not_completed(self):
        previous=signal.getsignal(signal.SIGTERM)
        with tempfile.TemporaryDirectory() as directory:
            counter=InteractionCounter(Path(directory)/'interactions.json')
            env=CountedEnv(FakeEnv(),counter,'validation')
            try:
                env.step(None);env.unwrapped.fail=True
                with self.assertRaises(RuntimeError): env.step(None)
                counter.flush()
                result=json.loads(counter.path.read_text())
                self.assertEqual(result['total_completed'],3)
                self.assertEqual(result['total_attempted'],6)
            finally:
                atexit.unregister(counter.flush)
                signal.signal(signal.SIGTERM,previous)


if __name__=='__main__': unittest.main()
