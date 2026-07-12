# ---------------------------------------------------------------------------
# Long-Task Agent  ·  the demo's PULL agent (A2A tasks/get + tasks/cancel)
# ---------------------------------------------------------------------------
# The Research Agent (in this same repo) shows the PUSH model: the agent
# calls the client back over a webhook. This agent shows the other half of
# A2A's async story — the PULL model, where the CLIENT decides when to check
# in, and can also change its mind and cancel:
#
#   message/send (blocking=false)  ->  Task "submitted"; work starts in the
#                                       background, same as the Research Agent
#   tasks/get    {id}              ->  a snapshot of the Task RIGHT NOW.
#                                       Call it as many times as you like —
#                                       each call is a fresh, independent
#                                       request/response; the agent does not
#                                       track "who's asking" or "how often."
#   tasks/cancel {id}              ->  ask the agent to stop. It cancels the
#                                       background asyncio.Task and the state
#                                       becomes "canceled" — a *terminal*
#                                       state, same family as "completed" or
#                                       "failed". You cannot cancel (or
#                                       re-cancel) a task that's already
#                                       terminal; the agent replies with the
#                                       spec's TaskNotCancelableError (-32002).
#
# No LLM, no real work here on purpose — just asyncio.sleep(seconds) — so the
# polling/cancellation mechanics are the whole story, with nothing else in
# the way.
# ---------------------------------------------------------------------------

import re
import asyncio
import datetime
import uuid

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="Long-Task Agent")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

AGENT_CARD = {
    "protocolVersion": "1.0",
    "name": "Long-Task Agent",
    "description": "Runs a long background job (just asyncio.sleep) so a client can poll it "
                    "with tasks/get and stop it early with tasks/cancel.",
    "url": "http://long-task-agent:8102/",
    "version": "1.0.0",
    "preferredTransport": "JSONRPC",
    # Deliberately NOT pushNotifications — this agent is the "you have to ask me" agent.
    "capabilities": {"streaming": False, "pushNotifications": False},
    "skills": [{
        "id": "long-sleep",
        "name": "Long background job",
        "description": "Sleeps for N seconds in the background. Poll tasks/get for status, "
                        "or tasks/cancel to stop it early.",
        "tags": ["demo", "async"],
        "examples": ["Sleep for 60 seconds"],
    }],
}

# taskId -> {"contextId", "status", "metadata", "_asyncio_task"}
TASKS: dict[str, dict] = {}


def _now() -> str:
    return datetime.datetime.utcnow().isoformat() + "Z"


def _snapshot(tid: str) -> dict:
    t = TASKS[tid]
    return {"id": tid, "contextId": t["contextId"], "status": t["status"], "metadata": t.get("metadata", {})}


async def _run(tid: str, seconds: float) -> None:
    TASKS[tid]["status"] = {"state": "working", "timestamp": _now()}
    try:
        await asyncio.sleep(seconds)
    except asyncio.CancelledError:
        TASKS[tid]["status"] = {"state": "canceled", "timestamp": _now()}
        raise
    TASKS[tid]["status"] = {
        "state": "completed", "timestamp": _now(),
        "message": {"role": "agent", "messageId": "m-" + uuid.uuid4().hex[:6],
                    "parts": [{"text": f"slept {seconds:.0f}s without interruption"}]},
    }


def _text_of(body: dict) -> str:
    return (((body.get("params") or {}).get("message") or {}).get("parts") or [{}])[0].get("text", "")


def _parse_seconds(text: str) -> float:
    m = re.search(r"(\d+(\.\d+)?)", text or "")
    return float(m.group(1)) if m else 30.0


@app.get("/.well-known/agent-card.json")
def agent_card():
    return AGENT_CARD


@app.get("/health")
def health():
    return {"ok": True}


@app.get("/debug/tasks/{task_id}")
def debug_task(task_id: str):
    if task_id not in TASKS:
        return {"error": "not found"}
    return _snapshot(task_id)


@app.post("/")
async def rpc(request: Request):
    body = await request.json()
    rpc_id = body.get("id")
    method = body.get("method")
    params = body.get("params") or {}

    # ---- message/send : accept the job, optionally non-blocking -----------
    if method == "message/send":
        text = _text_of(body)
        seconds = _parse_seconds(text)
        blocking = (params.get("configuration") or {}).get("blocking", True)

        tid, cid = "task-" + uuid.uuid4().hex[:6], "ctx-" + uuid.uuid4().hex[:6]
        TASKS[tid] = {"contextId": cid, "status": {"state": "submitted", "timestamp": _now()},
                      "metadata": {"seconds": seconds}}
        TASKS[tid]["_asyncio_task"] = asyncio.create_task(_run(tid, seconds))

        if not blocking:
            return {"jsonrpc": "2.0", "id": rpc_id, "result": _snapshot(tid)}

        try:                                    # blocking fallback: actually wait it out
            await TASKS[tid]["_asyncio_task"]
        except asyncio.CancelledError:
            pass
        return {"jsonrpc": "2.0", "id": rpc_id, "result": _snapshot(tid)}

    # ---- tasks/get : a poll. Stateless from the caller's point of view ----
    if method == "tasks/get":
        tid = params.get("id")
        if tid not in TASKS:
            return {"jsonrpc": "2.0", "id": rpc_id, "error": {"code": -32001, "message": "task not found"}}
        return {"jsonrpc": "2.0", "id": rpc_id, "result": _snapshot(tid)}

    # ---- tasks/cancel : client-initiated stop ------------------------------
    if method == "tasks/cancel":
        tid = params.get("id")
        if tid not in TASKS:
            return {"jsonrpc": "2.0", "id": rpc_id, "error": {"code": -32001, "message": "task not found"}}
        state = TASKS[tid]["status"]["state"]
        if state in ("completed", "canceled", "failed"):
            return {"jsonrpc": "2.0", "id": rpc_id,
                    "error": {"code": -32002, "message": f"task cannot be canceled in state {state}"}}
        bg = TASKS[tid]["_asyncio_task"]
        bg.cancel()
        try:                                     # wait for _run's except block to actually land
            await bg
        except asyncio.CancelledError:
            pass
        return {"jsonrpc": "2.0", "id": rpc_id, "result": _snapshot(tid)}

    return {"jsonrpc": "2.0", "id": rpc_id, "error": {"code": -32601, "message": "method not found"}}
