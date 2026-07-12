# ---------------------------------------------------------------------------
# Booking Agent  ·  the demo's MULTI-TURN (input-required) agent
# ---------------------------------------------------------------------------
# Booking is the canonical *stateful* task: the client sends what it has, the
# agent notices something is missing and pauses the task at `input-required`,
# the client answers ON THE SAME taskId, and the task resumes to `completed`
# with a PNR artifact. That is the whole lesson — a Task that lives across
# turns, threaded by taskId / contextId, held in server-side state.
#
# Same brain as every agent: SENSE -> REASON extracts the booking fields the
# user gave THIS turn; ACT merges them into the task's accumulated fields and
# either asks for what's still missing, or issues the booking.
# ---------------------------------------------------------------------------

import os
import uuid

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware

import llm
import booking_api

app = FastAPI(title="Booking Agent")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# what a complete booking needs. Turn 1 usually has passenger+flight; seat and
# dateOfBirth are what force the input-required pause.
REQUIRED = ["passenger", "flight", "seat", "dateOfBirth"]

# server-side Task state, keyed by taskId — this is what makes the task stateful
TASKS: dict = {}

AGENT_CARD = {
    "protocolVersion": "1.0",
    "name": "Booking Agent",
    "description": "Books a flight over a multi-turn task, asking for missing details.",
    "url": "http://flights-book-agent:8005/",
    "version": "1.0.0",
    "preferredTransport": "JSONRPC",
    "additionalInterfaces": [
        {"url": "http://flights-book-agent:8005/", "transport": "JSONRPC"},
    ],
    "capabilities": {"streaming": False},
    "skills": [{
        "id": "book-flight",
        "name": "Book a flight",
        "description": "Reserve a seat; asks for passenger, seat and date of birth as needed.",
        "tags": ["travel", "flights", "booking"],
        "examples": ["Book flight BA212 for John Smith"],
    }],
}

SYSTEM = ("You are the Booking Agent. From the message, extract any of these the user actually "
          "gave THIS turn: passenger (full name), flight (flight number like BA212), "
          "seat (aisle or window), dateOfBirth (YYYY-MM-DD). Call book_flight with only those "
          "fields — omit anything not provided. Never invent values.")

TOOL = {
    "name": "book_flight",
    "description": "Submit the booking fields the user provided this turn.",
    "parameters": {
        "type": "object",
        "properties": {
            "passenger": {"type": "string", "description": "passenger full name"},
            "flight": {"type": "string", "description": "flight number, e.g. BA212"},
            "seat": {"type": "string", "description": "seat preference: aisle or window"},
            "dateOfBirth": {"type": "string", "description": "date of birth YYYY-MM-DD"},
        },
        "required": [],
    },
}


def _make_run_tool(task: dict):
    """ACT target bound to THIS task: merge the newly-extracted fields into the
    task's accumulated state, then decide — still missing something, or book?"""
    async def run_tool(name, args):
        for k in REQUIRED:
            v = args.get(k)
            if v:
                task["fields"][k] = str(v).strip()
        missing = [k for k in REQUIRED if not task["fields"].get(k)]
        if missing:
            return {"status": "input-required", "missing": missing, "have": dict(task["fields"])}
        result = await booking_api.book_flight(task["fields"])
        return {"status": "completed", **result}
    return run_tool


@app.get("/.well-known/agent-card.json")
def agent_card():
    return AGENT_CARD


@app.get("/health")
def health():
    return {"ok": True}


@app.post("/")
async def rpc(request: Request):
    body = await request.json()
    rpc_id = body.get("id")
    if body.get("method") != "message/send":
        return {"jsonrpc": "2.0", "id": rpc_id,
                "error": {"code": -32601, "message": "method not found"}}

    params = body.get("params") or {}
    msg = params.get("message") or {}
    text = (msg.get("parts") or [{}])[0].get("text", "")

    # continue an existing task if the client threaded a taskId, else start one
    task_id = msg.get("taskId") or params.get("taskId")
    if task_id and task_id in TASKS:
        task = TASKS[task_id]
        context_id = task["contextId"]
    else:
        task_id = "task-" + uuid.uuid4().hex[:6]
        context_id = "ctx-" + uuid.uuid4().hex[:6]
        task = {"contextId": context_id, "fields": {}}
        TASKS[task_id] = task

    # SENSE -> REASON (extract fields) -> ACT (merge + decide) -> phrase the reply
    final_text, tool_result, trace = await llm.run(SYSTEM, text, TOOL, _make_run_tool(task))
    status = (tool_result or {}).get("status") or "input-required"

    if status == "input-required":
        # pause the task and ask; keep the state under task_id for the next turn
        result = {
            "id": task_id, "contextId": context_id,
            "status": {"state": "input-required",
                       "message": {"role": "agent", "messageId": "m-" + uuid.uuid4().hex[:6],
                                   "parts": [{"text": final_text}]}},
            "metadata": {"provider": llm.provider_label(), "trace": trace,
                         "missing": (tool_result or {}).get("missing"),
                         "have": (tool_result or {}).get("have")},
        }
    else:
        TASKS.pop(task_id, None)   # terminal state — free the server-side state
        result = {
            "id": task_id, "contextId": context_id, "status": {"state": "completed"},
            "artifacts": [{"artifactId": "art-" + uuid.uuid4().hex[:6], "name": "booking",
                           "parts": [{"text": final_text}, {"data": tool_result}]}],
            "metadata": {"provider": llm.provider_label(), "trace": trace},
        }

    return {"jsonrpc": "2.0", "id": rpc_id, "result": result}
