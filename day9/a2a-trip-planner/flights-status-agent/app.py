# ---------------------------------------------------------------------------
# Flight Status Agent  ·  the demo's message/stream (SSE) agent
# ---------------------------------------------------------------------------
# Flight status changes over time, so this agent answers with A2A's STREAMING
# shape: one request, an open connection, a sequence of Server-Sent Events.
#
#   message/stream  ->  POST /  (method "message/stream")  ->  text/event-stream
#   message/send    ->  POST /  (method "message/send")    ->  one Task snapshot
#
# Same brain as every other agent: SENSE -> REASON picks get_flight_status; then
# ACT *streams* the tool's updates. Each update is wrapped as a JSON-RPC
# response carrying a TaskStatusUpdateEvent (the A2A streaming result type).
# capabilities.streaming = true advertises this on the card.
# ---------------------------------------------------------------------------

import os
import json
import uuid
import datetime

from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware

import llm
from status_api import stream_status

app = FastAPI(title="Flight Status Agent")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

AGENT_CARD = {
    "protocolVersion": "1.0",
    "name": "Flight Status Agent",
    "description": "Streams a flight's live status from scheduled to landed.",
    "url": "http://flights-status-agent:8004/",
    "version": "1.0.0",
    "preferredTransport": "JSONRPC",
    "additionalInterfaces": [
        {"url": "http://flights-status-agent:8004/", "transport": "JSONRPC"},
    ],
    # THE point of this agent: it advertises streaming, so clients know they can
    # open a message/stream and receive Server-Sent Events.
    "capabilities": {"streaming": True},
    "skills": [{
        "id": "get-flight-status",
        "name": "Track flight status",
        "description": "Follow a flight live through boarding, departure, en route and landing.",
        "tags": ["travel", "flights", "status"],
        "examples": ["What's the status of BA212?", "Track flight DL447"],
    }],
}

SYSTEM = ("You are the Flight Status Agent. Pull the flight number (like BA212 or DL447) "
          "out of the request and call get_flight_status with it. If a date is named "
          "(YYYY-MM-DD), pass it too; otherwise omit it.")

TOOL = {
    "name": "get_flight_status",
    "description": "Track a flight's live status by its flight number (IATA, e.g. BA212).",
    "parameters": {
        "type": "object",
        "properties": {
            "flightNumber": {"type": "string", "description": "IATA flight number, e.g. BA212"},
            "date": {"type": "string", "description": "optional date YYYY-MM-DD"},
        },
        "required": ["flightNumber"],
    },
}


async def stream_tool(name, args):
    """ACT target: an async generator so the loop can stream the updates out."""
    async for update in stream_status(args.get("flightNumber", ""), args.get("date", "")):
        yield update


@app.get("/.well-known/agent-card.json")
def agent_card():
    return AGENT_CARD


@app.get("/health")
def health():
    return {"ok": True}


# ---- A2A result builders --------------------------------------------------
def _now():
    return datetime.datetime.utcnow().isoformat() + "Z"


def _status_update(tid, cid, state, text, final, phase=None, data=None, extra=None):
    """A TaskStatusUpdateEvent — the result type A2A streams over SSE."""
    parts = [{"text": text}]
    if data:
        parts.append({"data": data})
    result = {
        "taskId": tid, "contextId": cid, "kind": "status-update", "final": final,
        "status": {"state": state, "timestamp": _now(),
                   "message": {"role": "agent", "messageId": "m-" + uuid.uuid4().hex[:6], "parts": parts}},
        "metadata": {},
    }
    if phase:
        result["metadata"]["phase"] = phase
    if extra:
        result["metadata"].update(extra)
    return result


def _sse(rpc_id, result):
    return "data: " + json.dumps({"jsonrpc": "2.0", "id": rpc_id, "result": result}) + "\n\n"


def _text_of(body):
    return (((body.get("params") or {}).get("message") or {}).get("parts") or [{}])[0].get("text", "")


@app.post("/")
async def rpc(request: Request):
    body = await request.json()
    method = body.get("method")
    rpc_id = body.get("id")
    text = _text_of(body)

    # ---- message/stream : the SSE showpiece -------------------------------
    if method == "message/stream":
        async def gen():
            tid, cid = "task-" + uuid.uuid4().hex[:6], "ctx-" + uuid.uuid4().hex[:6]
            async for ev in llm.run_stream(SYSTEM, text, TOOL, stream_tool):
                if ev["type"] == "open":
                    flight = ev["input"].get("flightNumber", "?")
                    yield _sse(rpc_id, _status_update(
                        tid, cid, "submitted", f"Tracking flight {flight}\u2026", False,
                        phase="Submitted", extra={"provider": llm.provider_label(), "trace": ev["trace"]}))
                elif ev["type"] == "update":
                    u = ev["update"]
                    yield _sse(rpc_id, _status_update(
                        tid, cid, u["state"], u["detail"], u.get("final", False),
                        phase=u.get("phase"), data=(u if u.get("final") else None),
                        extra={"source": u.get("source")}))
                elif ev["type"] == "text":
                    yield _sse(rpc_id, _status_update(
                        tid, cid, "completed", ev["text"], True,
                        phase="Completed", extra={"trace": ev.get("trace")}))
        return StreamingResponse(gen(), media_type="text/event-stream")

    # ---- message/send : same task, non-streaming snapshot -----------------
    if method == "message/send":
        tid, cid = "task-" + uuid.uuid4().hex[:6], "ctx-" + uuid.uuid4().hex[:6]
        final_u, trace = None, []
        async for ev in llm.run_stream(SYSTEM, text, TOOL, stream_tool):
            if ev["type"] == "open":
                trace = ev["trace"]
            elif ev["type"] == "update":
                final_u = ev["update"]
            elif ev["type"] == "text":
                trace = ev.get("trace", trace)
        task = {
            "id": tid, "contextId": cid, "status": {"state": "completed"},
            "artifacts": [{"artifactId": "art-" + uuid.uuid4().hex[:6], "name": "flight-status",
                           "parts": [{"text": (final_u or {}).get("detail", "done")}, {"data": final_u or {}}]}],
            "metadata": {"provider": llm.provider_label(), "trace": trace},
        }
        return {"jsonrpc": "2.0", "id": rpc_id, "result": task}

    return {"jsonrpc": "2.0", "id": rpc_id, "error": {"code": -32601, "message": "method not found"}}
