from __future__ import annotations
import argparse
import json
from pathlib import Path

from .core import ROOT, Store, atomic_json, code_commit, file_hash, run_round


def main():
    parser = argparse.ArgumentParser(description="真实实验执行器；下一轮研究决定来自独立网页审查")
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("make-plan")
    p.add_argument("--round", required=True)
    p.add_argument("--experiment", required=True, choices=["env_smoke", "bootstrap_probe", "ref_sac"])
    p.add_argument("--steps", type=int, default=65536)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--benchmark", default="MT1", choices=["MT1", "MT10", "MT50"])
    p.add_argument("--task", default="reach-v3")
    p.add_argument("--expert-trajectories", type=int, default=10)
    p.add_argument("--bc-updates", type=int, default=1000)
    p.add_argument("--eval-episodes", type=int, default=3)
    p.add_argument("--parent")
    r = sub.add_parser("run")
    r.add_argument("plan", type=Path)
    r.add_argument("--retry", action="store_true")
    sub.add_parser("status")
    pull = sub.add_parser("pull-review")
    pull.add_argument("--commit", required=True)
    pull.add_argument("--path", required=True)
    args = parser.parse_args()
    if args.command == "make-plan":
        plan = {"round_id": args.round, "experiment": args.experiment, "seed": args.seed,
                "env_transitions": args.steps, "num_envs": 1 if args.benchmark == "MT1" else int(args.benchmark[2:]),
                "timeout_seconds": 1800, "base_code_commit": code_commit(),
                "charter_sha256": file_hash(ROOT / "configs/charter.json"),
                "parent_round_id": args.parent,
                "parent_manifest_sha256": file_hash(ROOT / "reports" / args.parent / "manifest.json") if args.parent else None,
                "config": {"benchmark": args.benchmark, "task": args.task,
                           "expert_trajectories": args.expert_trajectories, "bc_updates": args.bc_updates,
                           "eval_episodes": args.eval_episodes}}
        path = ROOT / "plans" / f"{args.round}.json"
        if path.exists(): raise ValueError("计划已存在；不覆盖")
        atomic_json(path, plan)
        print(path)
    elif args.command == "run":
        plan = json.loads(args.plan.read_text())
        store = Store()
        store.register(plan)
        print(json.dumps(run_round(plan["round_id"], retry=args.retry), ensure_ascii=False))
    elif args.command == "status":
        print(json.dumps(Store().status(), ensure_ascii=False, indent=2))
    elif args.command == "pull-review":
        from .handoff import pull_review
        print(json.dumps(pull_review(args.commit, args.path), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
