# ---------------------------------------------------------------------------
# Research Agent  ·  the demo's FIRE-AND-FORGET agent (A2A push notifications)
# ---------------------------------------------------------------------------
# Deep research on a destination takes a while — too long for a client to sit
# on an open HTTP connection or an SSE stream. A2A's answer isn't "the client
# polls tasks/get in a loop"; it's PUSH NOTIFICATIONS:
#
#   1. the client's message/send carries configuration.blocking = false AND
#      configuration.pushNotificationConfig = { url, token }
#   2. this agent returns THE INSTANT it has accepted the work — a Task in
#      state "submitted", with no artifact yet (that's the "forget" half)
#   3. it keeps working in an asyncio background task, completely decoupled
#      from that first HTTP request/response
#   4. when done, it POSTs the finished Task (status=completed + Artifact) to
#      the caller's webhook URL, carrying back the caller's own token in an
#      X-A2A-Notification-Token header so the receiver can verify it's genuine
#
# No open connection ties the two ends together at any point after step 2 —
# the orchestrator could restart, the agent could take five minutes, and the
# handoff still works. That's the whole point of "fire and forget."
#
# If the caller does NOT ask for push notifications (no webhook, or
# configuration.blocking left true), this agent just falls back to the normal
# blocking message/send every other agent in this repo uses — same brain,
# same llm.py, it only *also* knows how to work asynchronously.
# ---------------------------------------------------------------------------

import os
import asyncio
import datetime
import uuid

import httpx
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware

import llm

app = FastAPI(title="Research Agent")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# Simulated "deep work" duration for the demo — split across 4 phases below.
RESEARCH_SECONDS = float(os.getenv("RESEARCH_SECONDS", "12"))

AGENT_CARD = {
    "protocolVersion": "1.0",
    "name": "Research Agent",
    "description": "Does slow, multi-step destination research and reports back when done.",
    "url": "http://research-agent:8101/",
    "version": "1.0.0",
    "preferredTransport": "JSONRPC",
    "additionalInterfaces": [
        {"url": "http://research-agent:8101/", "transport": "JSONRPC"},
    ],
    # THE point of this agent: it can accept work non-blocking and push the
    # result to a webhook instead of the caller waiting or streaming.
    "capabilities": {"streaming": False, "pushNotifications": True},
    "skills": [{
        "id": "deep-research",
        "name": "Deep destination research",
        "description": "Research a multi-day itinerary sketch for a destination. Slow — built for "
                        "non-blocking send with a pushNotificationConfig webhook.",
        "tags": ["travel", "research", "async"],
        "examples": ["Research a 4-day itinerary for Kyoto in spring, and notify me when it's ready."],
    }],
}

SYSTEM = ("You are the Research Agent. Call deep_research with the destination and number of days. "
          "Then, using the researchNotes it returns, write a short multi-day itinerary sketch "
          "(2-3 sentences per day) as your final answer. Do not invent facts beyond the notes.")

TOOL = {
    "name": "deep_research",
    "description": "Slowly research a destination across several angles and return raw notes.",
    "parameters": {
        "type": "object",
        "properties": {
            "destination": {"type": "string", "description": "city or region, e.g. Kyoto"},
            "days": {"type": "integer", "description": "trip length in days"},
        },
        "required": ["destination"],
    },
}

# in-memory task store — for the /debug endpoint only, not part of the A2A flow
TASKS: dict[str, dict] = {}


def _now() -> str:
    return datetime.datetime.utcnow().isoformat() + "Z"


async def run_tool(name, args):
    """ACT — the 'slow' part of the demo: a few research phases, each with a
    real delay, so the fire-and-forget gap is visible rather than instant."""
    destination = args.get("destination") or "the destination"
    days = int(args.get("days") or 3)
    phases = [
        "scanning neighborhoods and seasonal weather notes",
        "checking festivals / events in the travel window",
        "cross-referencing transit and walking distances",
        "pacing a realistic day-by-day rhythm",
    ]
    per_phase = max(RESEARCH_SECONDS / len(phases), 0)
    notes = []
    for phase in phases:
        await asyncio.sleep(per_phase)
        notes.append(phase)
    return {"destination": destination, "days": days, "researchNotes": notes}


def _text_of(body: dict) -> str:
    return (((body.get("params") or {}).get("message") or {}).get("parts") or [{}])[0].get("text", "")


def _push_config(body: dict) -> dict:
    return (body.get("params") or {}).get("configuration") or {}


async def _notify_webhook(task: dict, url: str, token: str | None, attempts: int = 3) -> None:
    """Deliver the finished Task to the caller's webhook. At-least-once, with
    backoff — the caller's endpoint may be briefly unavailable; that's normal
    for webhooks, unlike a held-open stream where the caller must be there."""
    headers = {"Content-Type": "application/json"}
    if token:
        headers["X-A2A-Notification-Token"] = token
    delay = 1.0
    async with httpx.AsyncClient(timeout=15.0) as client:
        for attempt in range(1, attempts + 1):
            try:
                resp = await client.post(url, json=task, headers=headers)
                resp.raise_for_status()
                return
            except httpx.HTTPError:
                if attempt == attempts:
                    return
                await asyncio.sleep(delay)
                delay *= 2


async def _background_research(tid: str, cid: str, text: str, webhook: str, token: str | None) -> None:
    TASKS[tid] = {"id": tid, "status": {"state": "working"}}
    final_text, data, trace = await llm.run(SYSTEM, text, TOOL, run_tool)
    task = {
        "id": tid, "contextId": cid, "status": {"state": "completed", "timestamp": _now()},
        "artifacts": [{"artifactId": "art-" + uuid.uuid4().hex[:6], "name": "itinerary-research",
                       "parts": [{"text": final_text}, {"data": data or {}}]}],
        "metadata": {"provider": llm.provider_label(), "trace": trace, "deliveredAt": _now()},
    }
    TASKS[tid] = task
    await _notify_webhook(task, webhook, token)


@app.get("/.well-known/agent-card.json")
def agent_card():
    return AGENT_CARD


@app.get("/health")
def health():
    return {"ok": True}


@app.get("/debug/tasks/{task_id}")
def debug_task(task_id: str):
    """Not part of A2A — lets you watch this agent's own state flip
    submitted -> working -> completed, independent of the webhook delivery."""
    return TASKS.get(task_id, {"error": "not found"})


@app.post("/")
async def rpc(request: Request):
    body = await request.json()
    rpc_id = body.get("id")
    if body.get("method") != "message/send":
        return {"jsonrpc": "2.0", "id": rpc_id, "error": {"code": -32601, "message": "method not found"}}

    text = _text_of(body)
    config = _push_config(body)
    push = config.get("pushNotificationConfig") or {}
    webhook_url = push.get("url")
    token = push.get("token")
    blocking = config.get("blocking", True)

    tid, cid = "task-" + uuid.uuid4().hex[:6], "ctx-" + uuid.uuid4().hex[:6]

    # ---- fire-and-forget path: ack now, work in the background, push later ----
    if not blocking and webhook_url:
        TASKS[tid] = {"id": tid, "status": {"state": "submitted"}}
        asyncio.create_task(_background_research(tid, cid, text, webhook_url, token))
        return {
            "jsonrpc": "2.0", "id": rpc_id,
            "result": {
                "id": tid, "contextId": cid,
                "status": {"state": "submitted", "timestamp": _now()},
                "metadata": {
                    "provider": llm.provider_label(),
                    "note": "accepted — working in the background; result will be POSTed to "
                            "your pushNotificationConfig.url",
                },
            },
        }

    # ---- fallback: ordinary blocking message/send, same as any other agent ----
    final_text, data, trace = await llm.run(SYSTEM, text, TOOL, run_tool)
    task = {
        "id": tid, "contextId": cid, "status": {"state": "completed"},
        "artifacts": [{"artifactId": "art-" + uuid.uuid4().hex[:6], "name": "itinerary-research",
                       "parts": [{"text": final_text}, {"data": data or {}}]}],
        "metadata": {"provider": llm.provider_label(), "trace": trace},
    }
    TASKS[tid] = task
    return {"jsonrpc": "2.0", "id": rpc_id, "result": task}
