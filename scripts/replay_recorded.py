"""Run the minimally repaired official replay with explicit interaction counts.

CPU conversion uses two environments: original controller and target controller.
Count both. Multiprocess CPU replay is deliberately rejected until its per-worker
ledgers are supported; GPU vector replay counts actual padded vector steps too.
"""
import json
from pathlib import Path
import signal
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import gymnasium as gym
import tyro
from scripts import replay_trajectory as replay


def main():
    if len(sys.argv) < 3:
        raise SystemExit('Usage: replay_recorded.py COUNTER_PATH [official replay arguments]')
    counter_path = Path(sys.argv[1])
    args = tyro.cli(replay.Args, args=sys.argv[2:])
    if args.num_envs > 1 and args.sim_backend in (None, 'cpu', 'physx_cpu'):
        raise ValueError('Counted CPU replay currently requires --num-envs 1')
    if args.use_env_states:
        raise ValueError('Per-step state forcing cannot be used as real training transitions')
    counters = []
    original_make = gym.make

    def save(complete=False):
        counter_path.parent.mkdir(parents=True, exist_ok=True)
        temp = counter_path.with_suffix('.tmp')
        temp.write_text(json.dumps(dict(complete=complete, environments=counters,
            total_interactions=sum(x['interactions'] for x in counters), replay_args=vars(args)), indent=2))
        temp.replace(counter_path)

    class Counted(gym.Wrapper):
        def __init__(self, env, kwargs):
            super().__init__(env)
            self.record = dict(control_mode=kwargs.get('control_mode'), obs_mode=kwargs.get('obs_mode'),
                num_envs=env.unwrapped.num_envs, vector_step_calls=0, interactions=0)
            counters.append(self.record)

        def step(self, action):
            result = self.env.step(action)
            self.record['vector_step_calls'] += 1
            self.record['interactions'] += self.record['num_envs']
            if self.record['vector_step_calls'] % 1000 == 0:
                save()
            return result

    def counted_make(*positional, **kwargs):
        return Counted(original_make(*positional, **kwargs), kwargs)

    def stop(signum, frame):
        raise KeyboardInterrupt(f'Received {signum}')

    signal.signal(signal.SIGTERM, stop)
    complete = False
    gym.make = counted_make
    try:
        replay.main(args)
        complete = True
    finally:
        gym.make = original_make
        save(complete)


if __name__ == '__main__':
    main()
