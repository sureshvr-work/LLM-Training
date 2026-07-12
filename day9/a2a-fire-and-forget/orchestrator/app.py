# ---------------------------------------------------------------------------
# Orchestrator  ·  fires a non-blocking A2A request, then FORGETS about it
# ---------------------------------------------------------------------------
# POST /research/start   discovers the Research Agent, sends it a non-blocking
#                         message/send with our own /webhook/tasks URL as its
#                         pushNotificationConfig, and returns immediately with
#                         just an ack (taskId, state=submitted). No polling,
#                         no open connection to the sub-agent — that request
#                         is over.
#
# POST /webhook/tasks    the sub-agent calls THIS route, on its own schedule,
#                         whenever it finishes. This is the only place the
#                         result ever arrives — there is no other path.
#
# GET  /research/{id}/stream   an SSE relay from the orchestrator to the
#                         BROWSER only, so the demo UI updates the instant the
#                         webhook lands instead of polling. Don't confuse this
#                         with the orchestrator<->sub-agent leg above: that leg
#                         has zero open connections and zero polling. This SSE
#                         is purely "how do I make the browser update nicely."
#
# The second half of this file talks to the Long-Task Agent instead — the
# PULL model. POST /long/start kicks it off the same way (non-blocking
# message/send), but there's no webhook here: GET /long/{id} calls tasks/get
# fresh every time it's hit (a real poll, not a cache read), and
# POST /long/{id}/cancel calls tasks/cancel. Compare this to /research/* above
# to see push vs. pull side by side.
# ---------------------------------------------------------------------------

import os
import json
import time
import asyncio
import secrets
import datetime

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware

import a2a_client

app = FastAPI(title="Orchestrator")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

RESEARCH_URL = os.getenv("RESEARCH_URL", "http://research-agent:8101")
# The URL the sub-agent should call US back on. In docker-compose this is the
# service DNS name; in a real deployment it'd be your public webhook endpoint
# (e.g. behind ngrok/a load balancer) since the sub-agent may run anywhere.
SELF_URL = os.getenv("SELF_URL", "http://orchestrator:8103")
LONG_URL = os.getenv("LONG_URL", "http://long-task-agent:8102")

AGENT_CARD = {
    "protocolVersion": "1.0",
    "name": "Trip Research Orchestrator",
    "description": "Fires off deep destination research and reports back once the webhook delivers it.",
    "url": SELF_URL + "/",
    "version": "1.0.0",
    "preferredTransport": "JSONRPC",
    "capabilities": {"streaming": False},
    "skills": [{
        "id": "plan-research",
        "name": "Kick off destination research",
        "description": "Fire-and-forget a research request to the Research Agent.",
        "tags": ["travel", "async"],
        "examples": ["Research a 4-day trip to Kyoto"],
    }, {
        "id": "long-task",
        "name": "Poll / cancel a long background job",
        "description": "Start a long job on the Long-Task Agent, then poll or cancel it.",
        "tags": ["async", "demo"],
        "examples": ["Sleep for 60 seconds"],
    }],
}

# All in-memory — a real deployment would use a DB / queue so state survives restarts.
TASKS: dict[str, dict] = {}      # taskId -> latest known view
SECRETS: dict[str, str] = {}     # taskId -> webhook token we expect back
QUEUES: dict[str, "asyncio.Queue"] = {}   # taskId -> queue feeding the SSE relay


def _now() -> str:
    return datetime.datetime.utcnow().isoformat() + "Z"


def _card_summary(card: dict) -> dict:
    skill = (card.get("skills") or [{}])[0]
    return {
        "name": card.get("name"), "description": card.get("description"),
        "pushNotifications": (card.get("capabilities") or {}).get("pushNotifications", False),
        "skill": {"id": skill.get("id"), "examples": skill.get("examples", [])},
    }


@app.get("/.well-known/agent-card.json")
def agent_card():
    return AGENT_CARD


@app.get("/health")
def health():
    return {"ok": True}


@app.post("/research/start")
async def start(body: dict):
    destination = (body.get("destination") or "Kyoto").strip()
    days = int(body.get("days") or 4)
    text = f"Research a {days}-day itinerary for {destination}, and notify me when it's ready."

    trace = []
    card, _ = await a2a_client.discover(RESEARCH_URL, trace)
    if not (card.get("capabilities") or {}).get("pushNotifications"):
        return JSONResponse({"error": "Research Agent does not advertise pushNotifications"}, status_code=502)

    token = secrets.token_hex(16)          # per-task secret; verified on the inbound webhook
    webhook_url = SELF_URL.rstrip("/") + "/webhook/tasks"

    t0 = time.monotonic()
    result = await a2a_client.send_nonblocking(RESEARCH_URL, text, webhook_url, token, trace)
    ack_ms = round((time.monotonic() - t0) * 1000)

    if result.get("error") or not result.get("id"):
        return JSONResponse({"error": "sub-agent rejected the request", "raw": result}, status_code=502)

    tid = result["id"]
    SECRETS[tid] = token
    QUEUES[tid] = asyncio.Queue()
    TASKS[tid] = {
        "taskId": tid, "state": "submitted", "destination": destination, "days": days,
        "trace": trace, "submittedAt": _now(), "ackMs": ack_ms,
    }
    return {
        "taskId": tid, "state": "submitted", "ackMs": ack_ms,
        "note": f"orchestrator got its HTTP response back from the sub-agent in {ack_ms}ms — "
                f"the real research work (~seconds) is still running in the background",
        "trace": trace, "card": _card_summary(card),
    }


@app.post("/webhook/tasks")
async def webhook(request: Request):
    """The other end of the handshake. The sub-agent POSTs here, unprompted,
    whenever it finishes — could be seconds or minutes after /research/start
    returned. We verify the per-task token before trusting the payload."""
    task = await request.json()
    tid = task.get("id")
    token = request.headers.get("X-A2A-Notification-Token")

    if not tid or tid not in SECRETS:
        return JSONResponse({"error": "unknown taskId"}, status_code=404)
    if token != SECRETS[tid]:
        return JSONResponse({"error": "invalid push notification token"}, status_code=401)

    parts = (task.get("artifacts") or [{}])[0].get("parts", [])
    text = next((p["text"] for p in parts if "text" in p), None)
    data = next((p["data"] for p in parts if "data" in p), None)
    meta = task.get("metadata") or {}

    view = {
        **TASKS.get(tid, {}),
        "state": (task.get("status") or {}).get("state", "completed"),
        "text": text, "data": data,
        "resultTrace": meta.get("trace"), "provider": meta.get("provider"),
        "receivedAt": _now(),
    }
    TASKS[tid] = view

    queue = QUEUES.get(tid)
    if queue:
        await queue.put(view)
    return {"ok": True}


@app.get("/research/{task_id}")
async def get_task(task_id: str):
    return TASKS.get(task_id, {"error": "not found"})


@app.get("/research/{task_id}/stream")
async def stream_task(task_id: str):
    """SSE relay: orchestrator -> browser only (see module docstring)."""
    async def gen():
        queue = QUEUES.get(task_id)
        if queue is None:
            yield "data: " + json.dumps({"kind": "error", "message": "unknown taskId"}) + "\n\n"
            return
        yield "data: " + json.dumps({"kind": "ack", **TASKS.get(task_id, {})}) + "\n\n"
        while True:
            try:
                item = await asyncio.wait_for(queue.get(), timeout=4.0)
                yield "data: " + json.dumps({"kind": "webhook-delivered", **item}) + "\n\n"
                return
            except asyncio.TimeoutError:
                yield "data: " + json.dumps({"kind": "heartbeat", "waitingSince":
                                              TASKS.get(task_id, {}).get("submittedAt")}) + "\n\n"
    return StreamingResponse(gen(), media_type="text/event-stream")


# ===========================================================================
# PULL  ·  Long-Task Agent — non-blocking send, then WE poll and can cancel
# ===========================================================================
@app.post("/long/start")
async def long_start(body: dict):
    seconds = int(body.get("seconds") or 30)
    text = f"Sleep for {seconds} seconds"

    trace = []
    await a2a_client.discover(LONG_URL, trace)
    result = await a2a_client.send_async(LONG_URL, text, trace)
    if result.get("error") or not result.get("id"):
        return JSONResponse({"error": "sub-agent rejected the request", "raw": result}, status_code=502)

    tid = result["id"]
    return {"taskId": tid, "state": (result.get("status") or {}).get("state", "submitted"),
            "seconds": seconds, "trace": trace}


@app.get("/long/{task_id}")
async def long_poll(task_id: str):
    """A genuine poll — every call is a fresh tasks/get to the sub-agent,
    nothing cached on this side."""
    trace = []
    result = await a2a_client.get_task(LONG_URL, task_id, trace)
    if result.get("error"):
        return JSONResponse(result, status_code=404)
    status = result.get("status") or {}
    text = ((status.get("message") or {}).get("parts") or [{}])[0].get("text")
    return {"taskId": task_id, "state": status.get("state"), "text": text, "trace": trace}


@app.post("/long/{task_id}/cancel")
async def long_cancel(task_id: str):
    trace = []
    result = await a2a_client.cancel_task(LONG_URL, task_id, trace)
    if result.get("error"):
        return JSONResponse(result, status_code=409)
    status = result.get("status") or {}
    return {"taskId": task_id, "state": status.get("state"), "trace": trace}
