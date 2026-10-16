from __future__ import annotations

import contextlib
import datetime as dt
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import sqlite3
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS = {"env_smoke", "bootstrap_probe", "ref_sac"}
LOCKED_CONFIG = {"framework": "Meta-World", "version": "v3", "observation": "state",
                 "reward_function_version": "v2", "terminate_on_success": False}


def utcnow():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode()


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify_manifest(directory):
    directory = Path(directory).resolve()
    manifest = json.loads((directory / "manifest.json").read_text())
    if not isinstance(manifest, dict) or not manifest:
        raise ValueError("空证据清单")
    for name, expected in manifest.items():
        path = directory / name
        if not isinstance(name, str) or not path.resolve().is_relative_to(directory):
            raise ValueError("非法证据路径")
        if not path.is_file() or file_hash(path) != expected:
            raise ValueError(f"证据内容不匹配: {name}")


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    with temporary.open("w") as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(temporary, path)


def append_json(path, value):
    with Path(path).open("a") as f:
        f.write(canonical(value).decode() + "\n")
        f.flush()


def code_commit(root=ROOT):
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()


def validate_plan(plan, root=ROOT):
    expected = {"round_id", "experiment", "seed", "env_transitions", "num_envs", "timeout_seconds",
                "base_code_commit", "charter_sha256", "parent_round_id", "parent_manifest_sha256", "config"}
    if set(plan) != expected:
        raise ValueError(f"计划字段不匹配: missing={expected-set(plan)}, extra={set(plan)-expected}")
    if not re.fullmatch(r"R[0-9]{3,}", plan["round_id"]):
        raise ValueError("非法 round_id")
    if plan["experiment"] not in EXPERIMENTS:
        raise ValueError("不允许的实验；计划不能提交任意命令")
    for key, minimum, maximum in [("seed", 0, 2**31-1), ("env_transitions", 0, 1_000_000),
                                 ("num_envs", 1, 50), ("timeout_seconds", 1, 86400)]:
        if type(plan[key]) is not int or not minimum <= plan[key] <= maximum:
            raise ValueError(f"非法预算字段 {key}")
    if plan["env_transitions"] % plan["num_envs"]:
        raise ValueError("环境转移预算必须是并行环境数的倍数")
    if plan["base_code_commit"] != code_commit(root):
        raise ValueError("计划代码版本与当前 HEAD 不符")
    if plan["charter_sha256"] != file_hash(root / "configs/charter.json"):
        raise ValueError("研究章程发生变化")
    allowed = {"benchmark", "task", "learning_starts", "batch_size", "eval_freq", "eval_episodes",
               "bc_updates", "expert_trajectories", "training_freq", "utd", "bc_checkpoint", "dataset"}
    if not isinstance(plan["config"], dict) or set(plan["config"]) - allowed:
        raise ValueError("计划包含未批准配置")
    cfg = plan["config"]
    for key in ["bc_checkpoint", "dataset"]:
        if key in cfg:
            p = root / cfg[key]
            if not p.resolve().is_relative_to(root.resolve()):
                raise ValueError("数据和检查点必须位于项目目录")
    if cfg.get("benchmark", "MT1") not in {"MT1", "MT10", "MT50"}:
        raise ValueError("非法基准")
    if cfg.get("benchmark") in {"MT10", "MT50"} and plan["num_envs"] != int(cfg["benchmark"][2:]):
        raise ValueError("MT10/MT50 并行环境数应等于任务数")
    if "task" in cfg and not re.fullmatch(r"[a-z][a-z0-9-]+-v3", cfg["task"]):
        raise ValueError("非法任务名称")
    for key, lo, hi in [("learning_starts", 1, 100000), ("batch_size", 1, 2048),
                        ("eval_freq", 1, 1000000), ("eval_episodes", 1, 200),
                        ("bc_updates", 1, 100000), ("expert_trajectories", 1, 1000),
                        ("training_freq", 1, 1024)]:
        if key in cfg and (type(cfg[key]) is not int or not lo <= cfg[key] <= hi):
            raise ValueError(f"非法配置 {key}")
    if "utd" in cfg and (type(cfg["utd"]) not in {int, float} or not 0 < cfg["utd"] <= 1):
        raise ValueError("utd 必须在 (0,1] 内")
    if plan["experiment"] in {"ref_sac", "bc_sac", "u0"}:
        frequency = cfg.get("training_freq", 64)
        if frequency % plan["num_envs"] or plan["env_transitions"] % frequency:
            raise ValueError("训练预算与采样频率不整除，拒绝隐式超预算")
    parent = plan["parent_round_id"]
    if parent is not None:
        if not isinstance(parent, str) or not re.fullmatch(r"R[0-9]{3,}", parent):
            raise ValueError("非法父轮次")
        manifest = root / "reports" / parent / "manifest.json"
        if not manifest.is_file() or file_hash(manifest) != plan["parent_manifest_sha256"]:
            raise ValueError("父轮次证据不存在或哈希错误")
        verify_manifest(manifest.parent)
    elif plan["parent_manifest_sha256"] is not None:
        raise ValueError("无父轮次却包含父证据")


class Store:
    def __init__(self, root=ROOT):
        self.root = Path(root)
        (self.root / "runtime").mkdir(parents=True, exist_ok=True)
        self.path = self.root / "runtime/state.sqlite"
        with self.connect() as db:
            db.executescript("""
              CREATE TABLE IF NOT EXISTS rounds(
                id TEXT PRIMARY KEY, plan_hash TEXT NOT NULL, plan TEXT NOT NULL,
                status TEXT NOT NULL, started TEXT, finished TEXT, error TEXT,
                attempts INTEGER NOT NULL DEFAULT 0);
              CREATE TABLE IF NOT EXISTS events(
                seq INTEGER PRIMARY KEY, round_id TEXT, time TEXT NOT NULL,
                kind TEXT NOT NULL, payload TEXT NOT NULL);
              CREATE TABLE IF NOT EXISTS decisions(
                parent_round TEXT PRIMARY KEY, content_hash TEXT NOT NULL, payload TEXT NOT NULL);
            """)

    def connect(self):
        db = sqlite3.connect(self.path, timeout=30)
        db.row_factory = sqlite3.Row
        db.execute("PRAGMA journal_mode=WAL")
        return db

    def event(self, round_id, kind, payload):
        with self.connect() as db:
            db.execute("INSERT INTO events(round_id,time,kind,payload) VALUES(?,?,?,?)",
                       (round_id, utcnow(), kind, json.dumps(payload, ensure_ascii=False)))

    def register(self, plan):
        validate_plan(plan, self.root)
        key = digest(plan)
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            existing = db.execute("SELECT * FROM rounds WHERE id=?", (plan["round_id"],)).fetchone()
            if existing:
                if existing["plan_hash"] != key:
                    raise ValueError("同轮次已有不同计划，禁止覆盖")
                return False
            db.execute("INSERT INTO rounds(id,plan_hash,plan,status) VALUES(?,?,?,?)",
                       (plan["round_id"], key, canonical(plan).decode(), "QUEUED"))
        self.event(plan["round_id"], "registered", {"plan_sha256": key})
        return True

    def status(self):
        with self.connect() as db:
            return [dict(x) for x in db.execute("SELECT id,status,started,finished,error,attempts FROM rounds ORDER BY id")]

    def get(self, round_id):
        with self.connect() as db:
            row = db.execute("SELECT * FROM rounds WHERE id=?", (round_id,)).fetchone()
            if not row:
                raise ValueError("轮次未注册")
            return dict(row)

    def set_status(self, round_id, status, error=None):
        with self.connect() as db:
            db.execute("UPDATE rounds SET status=?,error=? WHERE id=?", (status, error, round_id))
        self.event(round_id, "status", {"status": status, "error": error})


@contextlib.contextmanager
def job_lock(root=ROOT):
    p = root / "runtime/job.lock"
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w") as f:
        try:
            fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError("已有实验进程持有作业锁") from None
        yield


def package_round(plan, root=ROOT):
    import shutil
    directory = root / "runs" / plan["round_id"]
    target = root / "reports" / plan["round_id"]
    if target.exists():
        return target
    staging = root / "reports" / (plan["round_id"] + ".tmp")
    staging.mkdir(parents=True, exist_ok=False)
    for name in ["plan.json", "metrics.jsonl", "result.json", "command.json", "exit.json", "environment.json"]:
        if (directory / name).is_file():
            shutil.copy2(directory / name, staging / name)
    result = json.loads((directory / "result.json").read_text()) if (directory / "result.json").exists() else {"error": "No result; see exit.json"}
    report = {"round_id": plan["round_id"], "synthetic": False,
              "code_commit": plan["base_code_commit"], "plan_sha256": digest(plan),
              "experiment": plan["experiment"], "result": result,
              "conclusion": "engineering_validation_only" if plan["experiment"] in {"env_smoke", "bootstrap_probe"} else "single_run_not_algorithm_evidence"}
    atomic_json(staging / "report.json", report)
    (staging / "REPORT.md").write_text(f"# {plan['round_id']} {plan['experiment']}\n\n"
        f"真实执行记录。代码 `{plan['base_code_commit']}`。\n\n"
        "本轮不能用于声明算法优于基线或具有新颖性。\n\n```json\n" +
        json.dumps(result, ensure_ascii=False, indent=2) + "\n```\n")
    atomic_json(staging / "manifest.json", {p.name: file_hash(p) for p in sorted(staging.iterdir()) if p.is_file()})
    os.replace(staging, target)
    return target


def run_round(round_id, root=ROOT, retry=False):
    store = Store(root)
    row = store.get(round_id)
    if row["status"] in {"WAIT_REVIEW", "REVIEWED", "COMPLETE"}:
        return {"status": row["status"], "duplicate_skipped": True}
    if row["status"] != "QUEUED" and not retry:
        raise ValueError("非排队状态；检查现场后显式 --retry，避免重复运行")
    plan = json.loads(row["plan"])
    validate_plan(plan, root)
    with job_lock(root):
        row = store.get(round_id)
        if row["status"] in {"WAIT_REVIEW", "REVIEWED", "COMPLETE"}:
            return {"status": row["status"], "duplicate_skipped": True}
        attempt = row["attempts"] + 1
        directory = root / "runs" / round_id
        if directory.exists():
            archive = root / "runs" / f"{round_id}.attempt-{attempt-1}"
            if archive.exists():
                raise ValueError("旧尝试归档已存在，拒绝覆盖证据")
            directory.rename(archive)
        directory.mkdir(parents=True)
        atomic_json(directory / "plan.json", plan)
        with store.connect() as db:
            db.execute("UPDATE rounds SET status='RUNNING',started=?,finished=NULL,error=NULL,attempts=? WHERE id=?",
                       (utcnow(), attempt, round_id))
        cmd = [sys.executable, "-u", "-m", "lab.tasks", str(directory / "plan.json")]
        env = os.environ.copy()
        env["MPLCONFIGDIR"] = str(root / "runtime/matplotlib")
        env["MUJOCO_GL"] = "egl"
        env["PYTHONUNBUFFERED"] = "1"
        atomic_json(directory / "command.json", {"argv": cmd, "cwd": str(root), "started": utcnow(), "attempt": attempt})
        started = time.monotonic()
        with (directory / "process.log").open("w") as log:
            proc = subprocess.Popen(cmd, cwd=root, env=env, stdout=log, stderr=subprocess.STDOUT,
                                    start_new_session=True)
            store.event(round_id, "process_started", {"pid": proc.pid, "attempt": attempt})
            timed_out = False
            try:
                while proc.poll() is None:
                    elapsed = time.monotonic() - started
                    atomic_json(root / "runtime/status.json", {"round_id": round_id, "stage": "RUNNING",
                                "pid": proc.pid, "elapsed_seconds": elapsed, "heartbeat": utcnow()})
                    if elapsed > plan["timeout_seconds"]:
                        timed_out = True
                        import signal
                        os.killpg(proc.pid, signal.SIGTERM)
                        try:
                            proc.wait(timeout=20)
                        except subprocess.TimeoutExpired:
                            os.killpg(proc.pid, signal.SIGKILL)
                        break
                    time.sleep(1)
                code = proc.wait()
            except BaseException:
                # Keep the process and lock relationship explicit; an interrupted controller
                # cannot safely leave an unowned GPU process running.
                import signal
                if proc.poll() is None:
                    os.killpg(proc.pid, signal.SIGTERM)
                    try: proc.wait(timeout=20)
                    except subprocess.TimeoutExpired: os.killpg(proc.pid, signal.SIGKILL); proc.wait()
                store.set_status(round_id, "INTERRUPTED", "controller interrupted; attempt preserved")
                raise
        atomic_json(directory / "exit.json", {"exit_code": code, "timed_out": timed_out,
                                               "seconds": time.monotonic()-started, "finished": utcnow()})
        with store.connect() as db:
            db.execute("UPDATE rounds SET finished=? WHERE id=?", (utcnow(), round_id))
        if code != 0:
            store.set_status(round_id, "FAILED", f"exit={code}; see runs/{round_id}/process.log")
            package_round(plan, root)
            return {"status": "FAILED", "exit_code": code}
        if not (directory / "result.json").exists():
            store.set_status(round_id, "FAILED", "missing result.json")
            raise RuntimeError("进程退出但没有结果文件")
        target = package_round(plan, root)
        store.set_status(round_id, "WAIT_REVIEW")
        atomic_json(root / "runtime/status.json", {"round_id": round_id, "stage": "WAIT_REVIEW", "heartbeat": utcnow()})
        return {"status": "WAIT_REVIEW", "report": str(target), "manifest_sha256": file_hash(target / "manifest.json")}
