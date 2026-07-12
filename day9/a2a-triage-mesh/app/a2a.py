"""
a2a.py — the A2A protocol pieces this mesh uses.

A2A is built on plain web standards: HTTP + JSON-RPC 2.0 + Server-Sent Events.
These helpers build the exact JSON shapes the spec defines, so every card,
request, and task below is genuinely curl-able.

Agent discovery is the star of this demo: each specialist serves an Agent Card
at /.well-known/agent-card.json, and the router LEARNS the capability graph by
reading those cards — nothing about who-handles-what is hardcoded.
"""
import json
import uuid


def new_id(prefix: str) -> str:
    return f"{prefix}-{uuid.uuid4().hex[:6]}"


# ── parts & messages ─────────────────────────────────────────────────────────
def text_part(text: str) -> dict:
    return {"kind": "text", "text": text}


def user_message(text: str, message_id: str | None = None) -> dict:
    return {"kind": "message", "role": "user",
            "messageId": message_id or new_id("msg"),
            "parts": [text_part(text)]}


def first_text(message: dict) -> str:
    for p in (message or {}).get("parts", []):
        if p.get("kind") == "text":
            return p.get("text", "")
    return ""


# ── tasks & artifacts ────────────────────────────────────────────────────────
def make_task(task_id, context_id, state, artifacts=None) -> dict:
    t = {"id": task_id, "contextId": context_id, "kind": "task",
         "status": {"state": state}}
    if artifacts is not None:
        t["artifacts"] = artifacts
    return t


def text_artifact(text, name="result", artifact_id=None) -> dict:
    return {"artifactId": artifact_id or new_id("art"), "name": name,
            "parts": [text_part(text)]}


# ── streaming event payloads (the `result` of each SSE frame) ────────────────
def status_event(task_id, context_id, state, final) -> dict:
    return {"taskId": task_id, "contextId": context_id,
            "kind": "status-update", "status": {"state": state}, "final": final}


def artifact_event(task_id, context_id, artifact) -> dict:
    return {"taskId": task_id, "contextId": context_id,
            "kind": "artifact-update", "artifact": artifact}


# ── JSON-RPC 2.0 envelopes ───────────────────────────────────────────────────
def rpc_result(req_id, result) -> dict:
    return {"jsonrpc": "2.0", "id": req_id, "result": result}


def rpc_error(req_id, code, message) -> dict:
    return {"jsonrpc": "2.0", "id": req_id, "error": {"code": code, "message": message}}


# ── SSE framing ──────────────────────────────────────────────────────────────
def sse(obj) -> str:
    return f"data: {json.dumps(obj)}\n\n"
