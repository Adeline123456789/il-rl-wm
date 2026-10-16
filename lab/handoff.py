"""Receive independent web-ChatGPT decisions; never generate/approve one here."""
from __future__ import annotations

import json
from pathlib import Path
import re
import urllib.request

from .core import ROOT, Store, atomic_json, digest, file_hash, validate_plan, verify_manifest

REPOSITORY = "Adeline123456789/il-rl-wm"


def extract_decision(response_text):
    candidates = re.findall(r"```(?:json)?\s*([\s\S]*?)```", response_text)
    candidates.append(response_text.strip())
    values = []
    for candidate in candidates:
        try:
            obj = json.loads(candidate)
        except (ValueError, TypeError):
            continue
        if isinstance(obj, dict) and "next_plan" in obj:
            if not any(digest(obj) == digest(v) for v in values): values.append(obj)
    if len(values) != 1: raise ValueError("网页响应必须包含唯一可解析的决定 JSON")
    return values[0]


def accept_review(packet, response_text, source, root=ROOT):
    required = {"parent_round_id", "parent_manifest_sha256", "reviewer", "conversation_url",
                "response_id", "evidence_read", "decision", "rationale", "next_plan"}
    required -= {"conversation_url", "response_id"}
    if set(packet) != required:
        raise ValueError("审查字段不完整或含未允许字段")
    if digest(extract_decision(response_text)) != digest(packet):
        raise ValueError("交接决定与网页原文不同，拒绝执行端编辑审批")
    if packet["reviewer"] != "web-chatgpt":
        raise ValueError("本地执行器不接受 Codex 自行审批")
    if not isinstance(source, dict) or set(source) != {"conversation_url", "response_id"}:
        raise ValueError("缺少网页来源回执")
    if not isinstance(source["conversation_url"], str) or not re.fullmatch(
            r"https://chatgpt\.com/c/[A-Za-z0-9-]+", source["conversation_url"]):
        raise ValueError("必须提供真实网页 ChatGPT 审查会话")
    if not isinstance(source["response_id"], str) or not source["response_id"]:
        raise ValueError("缺少网页助手响应标识")
    round_id = packet["parent_round_id"]
    if not isinstance(round_id, str) or not re.fullmatch(r"R[0-9]{3,}", round_id):
        raise ValueError("非法轮次")
    directory = root / "reports" / round_id
    verify_manifest(directory)
    if file_hash(directory / "manifest.json") != packet["parent_manifest_sha256"]:
        raise ValueError("审查证据版本不匹配")
    if not isinstance(packet["evidence_read"], list) or not packet["evidence_read"]:
        raise ValueError("审查端未确认读取证据")
    prefix = f"https://github.com/{REPOSITORY}/"
    if not all(isinstance(u, str) and u.startswith(prefix) for u in packet["evidence_read"]):
        raise ValueError("证据必须来自指定研究仓库")
    if packet["decision"] not in {"continue", "request_changes", "stop", "insufficient_evidence"}:
        raise ValueError("非法决定")
    next_plan = packet["next_plan"]
    if packet["decision"] == "continue":
        if not isinstance(next_plan, dict): raise ValueError("继续执行必须有下一计划")
        if next_plan["parent_round_id"] != round_id or next_plan["parent_manifest_sha256"] != packet["parent_manifest_sha256"]:
            raise ValueError("下一计划没有绑定当前证据")
        validate_plan(next_plan, root)
    elif next_plan is not None:
        raise ValueError("非继续决定不得包含可执行任务")
    store = Store(root)
    with store.connect() as db:
        db.execute("BEGIN IMMEDIATE")
        old = db.execute("SELECT content_hash FROM decisions WHERE parent_round=?", (round_id,)).fetchone()
        if old:
            if old["content_hash"] != digest(packet): raise ValueError("同轮次已有另一份审查，拒绝覆盖")
            return {"duplicate_skipped": True, "round_id": round_id}
        current = db.execute("SELECT status FROM rounds WHERE id=?", (round_id,)).fetchone()
        if not current or current["status"] != "WAIT_REVIEW":
            raise ValueError("只有等待审查的实验可以接收下一计划")
        db.execute("INSERT INTO decisions(parent_round,content_hash,payload) VALUES(?,?,?)",
                   (round_id, digest(packet), json.dumps(packet, ensure_ascii=False)))
        if next_plan is not None:
            existing = db.execute("SELECT plan_hash FROM rounds WHERE id=?", (next_plan["round_id"],)).fetchone()
            if existing and existing["plan_hash"] != digest(next_plan): raise ValueError("下一轮计划冲突")
            if not existing:
                db.execute("INSERT INTO rounds(id,plan_hash,plan,status) VALUES(?,?,?,?)",
                           (next_plan["round_id"], digest(next_plan), json.dumps(next_plan), "QUEUED"))
        receipt_dir = root / "reviews" / round_id
        receipt_dir.mkdir(parents=True, exist_ok=True)
        (receipt_dir / "web_response.txt").write_text(response_text)
        atomic_json(receipt_dir / "decision.json", packet)
        atomic_json(receipt_dir / "receipt.json", {
            "source_conversation": source["conversation_url"], "source_response_id": source["response_id"],
            "response_sha256": file_hash(receipt_dir / "web_response.txt"),
            "decision_sha256": digest(packet), "source": "web response relayed without editorial approval by executor",
            "provenance_limit": "Receipt records observed source; it is not a cryptographic attestation of model identity."})
        db.execute("UPDATE rounds SET status=? WHERE id=?", ("REVIEWED", round_id))
    if next_plan is not None:
        atomic_json(root / "plans" / f"{next_plan['round_id']}.json", next_plan)
    store.event(round_id, "web_review_received", {"decision": packet["decision"],
                "conversation_url": source["conversation_url"], "next_round": next_plan["round_id"] if next_plan else None})
    return {"review_received": True, "next_round": next_plan["round_id"] if next_plan else None,
            "next_status": "QUEUED" if next_plan else "WAIT_USER_OR_MORE_EVIDENCE"}


def pull_review(commit, path, root=ROOT):
    if not re.fullmatch(r"[a-f0-9]{40}", commit): raise ValueError("需要固定 Git 提交 SHA")
    if not re.fullmatch(r"exchange/R[0-9]{3,}\.json", path): raise ValueError("不允许的交接路径")
    url = f"https://raw.githubusercontent.com/{REPOSITORY}/{commit}/{path}"
    request = urllib.request.Request(url, headers={"User-Agent": "il-rl-wm-executor"})
    with urllib.request.urlopen(request, timeout=30) as response:
        raw = response.read(512001)
    if len(raw) > 512000: raise ValueError("审查信封过大")
    envelope = json.loads(raw)
    if set(envelope) != {"decision", "web_response", "source"}: raise ValueError("不完整的网页审查信封")
    return accept_review(envelope["decision"], envelope["web_response"], envelope["source"], root)
