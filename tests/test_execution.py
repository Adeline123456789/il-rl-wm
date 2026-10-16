import copy
import json
from concurrent.futures import ThreadPoolExecutor

import pytest

from lab import core


@pytest.fixture
def project(tmp_path, monkeypatch):
    (tmp_path / "configs").mkdir()
    (tmp_path / "configs/charter.json").write_text('{"teacher":"frozen"}')
    monkeypatch.setattr(core, "code_commit", lambda root=None: "a"*40)
    plan = {"round_id": "R000", "experiment": "bootstrap_probe", "seed": 0,
            "env_transitions": 65536, "num_envs": 1, "timeout_seconds": 100,
            "base_code_commit": "a"*40, "charter_sha256": core.file_hash(tmp_path / "configs/charter.json"),
            "parent_round_id": None, "parent_manifest_sha256": None,
            "config": {"benchmark": "MT1", "task": "reach-v3", "bc_updates": 1000}}
    return tmp_path, plan


def test_duplicate_delivery_is_idempotent(project):
    root, plan = project
    store = core.Store(root)
    assert store.register(plan) is True
    assert store.register(plan) is False
    assert len(store.status()) == 1


def test_concurrent_claim_cannot_duplicate(project):
    root, plan = project
    store = core.Store(root)
    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(lambda _: store.register(plan), range(4)))
    assert results.count(True) == 1


def test_conflicting_plan_rejected(project):
    root, plan = project
    store = core.Store(root); store.register(plan)
    changed = {**plan, "seed": 1}
    with pytest.raises(ValueError, match="不同计划"):
        store.register(changed)


@pytest.mark.parametrize("change", [
    {"base_code_commit": "b"*40}, {"env_transitions": 2000000},
    {"config": {"command": "arbitrary shell"}}, {"config": {"dataset": "../../outside"}},
    {"experiment": "unimplemented"}, {"seed": True},
])
def test_unapproved_execution_rejected(project, change):
    root, plan = project
    with pytest.raises(ValueError): core.validate_plan({**plan, **change}, root)


def test_payload_tampering_rejected(project):
    root, plan = project
    parent = root / "reports/R000"; parent.mkdir(parents=True)
    (parent / "metrics.json").write_text('{"success":0}')
    core.atomic_json(parent / "manifest.json", {"metrics.json": core.file_hash(parent / "metrics.json")})
    child = {**plan, "round_id": "R001", "parent_round_id": "R000",
             "parent_manifest_sha256": core.file_hash(parent / "manifest.json")}
    core.validate_plan(child, root)
    (parent / "metrics.json").write_text('{"success":1}')
    with pytest.raises(ValueError, match="证据内容"):
        core.validate_plan(child, root)


def test_frozen_teacher_not_updated():
    import torch
    from lab.models import Actor, SAC
    torch.set_num_threads(1)
    teacher = Actor(39, 4)
    agent = SAC(39, 4, "cpu", teacher=teacher, use_world_model=True)
    before = copy.deepcopy(agent.teacher.state_dict())
    batch = [torch.randn(8,39), torch.rand(8,4)*2-1, torch.rand(8,1), torch.randn(8,39), torch.zeros(8,1)]
    agent.update(batch)
    assert all(torch.equal(before[k], v) for k,v in agent.teacher.state_dict().items())
    assert all(not p.requires_grad for p in agent.teacher.parameters())
    calls = agent.teacher_calls
    agent.action(batch[0][0].numpy(), mode="no_il")
    assert agent.teacher_calls == calls


@pytest.fixture
def reviewed_round(project):
    root, plan = project
    store = core.Store(root); store.register(plan); store.set_status("R000", "WAIT_REVIEW")
    report = root / "reports/R000"; report.mkdir(parents=True)
    (report / "metrics.json").write_text('{"success":0}')
    core.atomic_json(report / "manifest.json", {"metrics.json": core.file_hash(report / "metrics.json")})
    mh = core.file_hash(report / "manifest.json")
    child = {**plan, "round_id": "R001", "experiment": "ref_sac", "parent_round_id": "R000",
             "parent_manifest_sha256": mh}
    packet = {"parent_round_id": "R000", "parent_manifest_sha256": mh, "reviewer": "web-chatgpt",
              "evidence_read": ["https://github.com/Adeline123456789/il-rl-wm/blob/pinned/reports/R000/metrics.json"],
              "decision": "continue", "rationale": "TEST ONLY: run a reference baseline", "next_plan": child}
    source = {"conversation_url": "https://chatgpt.com/c/test-fixture", "response_id": "test-message"}
    return root, packet, source


def test_only_external_decision_is_queued_and_never_executed(reviewed_round):
    from lab.handoff import accept_review
    root, packet, source = reviewed_round
    text = "```json\n" + json.dumps(packet) + "\n```"
    first = accept_review(packet, text, source, root)
    assert first["next_status"] == "QUEUED"
    assert core.Store(root).get("R001")["attempts"] == 0
    assert accept_review(packet, text, source, root)["duplicate_skipped"] is True
    assert (root / "reviews/R000/web_response.txt").read_text() == text


def test_executor_cannot_edit_web_decision(reviewed_round):
    from lab.handoff import accept_review
    root, packet, source = reviewed_round
    edited = {**packet, "rationale": "edited by executor"}
    with pytest.raises(ValueError, match="原文不同"):
        accept_review(edited, json.dumps(packet), source, root)


def test_codex_review_is_rejected(reviewed_round):
    from lab.handoff import accept_review
    root, packet, source = reviewed_round
    packet = {**packet, "reviewer": "codex"}
    with pytest.raises(ValueError, match="自行审批"):
        accept_review(packet, json.dumps(packet), source, root)
