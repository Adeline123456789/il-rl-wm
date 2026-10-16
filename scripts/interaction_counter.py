"""Count returned and attempted real env.step transitions, including interrupted runs."""
import atexit
import json
from pathlib import Path
import signal
import gymnasium as gym


class InteractionCounter:
    def __init__(self, path):
        self.path=Path(path)
        self.counts={}
        self.flush()
        atexit.register(self.flush)
        signal.signal(signal.SIGTERM,self.stop)

    def flush(self):
        temporary=self.path.with_suffix('.tmp')
        temporary.write_text(json.dumps(dict(phases=self.counts, total_completed=sum(v['completed_transitions'] for v in self.counts.values()), total_attempted=sum(v['attempted_transitions'] for v in self.counts.values()), definition='Completed counts only returned env.step calls; attempted minus completed flags an interrupted or failed step. No reset calls counted.'),indent=2))
        temporary.replace(self.path)

    def stop(self, signum, frame):
        self.flush()
        raise SystemExit(128+signum)


class CountedEnv(gym.Wrapper):
    def __init__(self, env, counter, phase):
        super().__init__(env)
        self.counter=counter
        self.phase=phase
        self.n=env.unwrapped.num_envs
        self.count=counter.counts.setdefault(phase,dict(attempted_transitions=0,completed_transitions=0,completed_vector_steps=0))

    def step(self, action):
        self.count['attempted_transitions']+=self.n
        result=self.env.step(action)
        self.count['completed_transitions']+=self.n
        self.count['completed_vector_steps']+=1
        if self.count['completed_vector_steps']%50==0:
            self.counter.flush()
        return result
