# ---------------------------------------------------------------------------
# Flights Agent  ·  a complete A2A agent in one file (identical shape to Weather)
# ---------------------------------------------------------------------------
#   GET  /.well-known/agent-card.json   -> discovery (REST)
#   POST /                              -> messages   (JSON-RPC 2.0)
# Its real tool is AviationStack. The LLM maps city names -> IATA codes.
# ---------------------------------------------------------------------------

import uuid
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

import llm
from flights_api import search_flights

app = FastAPI(title="Flights Agent")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

AGENT_CARD = {
    "protocolVersion": "1.0",
    "name": "Flights Agent",
    "description": "Finds real flights between two airports.",
    "url": "http://flights-agent:8002/",
    "version": "1.0.0",
    # This agent prefers the REST binding. Same Message/Task objects as JSON-RPC,
    # just a different envelope — it also offers JSON-RPC (see additionalInterfaces).
    "preferredTransport": "HTTP+JSON",
    "additionalInterfaces": [
        {"url": "http://flights-agent:8002/", "transport": "HTTP+JSON"},
        {"url": "http://flights-agent:8002/", "transport": "JSONRPC"},
    ],
    "capabilities": {"streaming": False},
    "skills": [{
        "id": "search-flights",
        "name": "Search flights",
        "description": "List flights between two cities/airports.",
        "tags": ["travel", "flights"],
        "examples": ["Find flights from Boston to Tokyo"],
    }],
}

SYSTEM = ("You are the Flights Agent. Convert the cities in the request to IATA airport "
          "codes and call search_flights. Then reply in one short sentence. "
          "Use the busiest international airport when a city has several (e.g. London -> LHR). "
          "If the request names a date (YYYY-MM-DD), pass it as departureDate; otherwise omit it.")

TOOL = {
    "name": "search_flights",
    "description": "Search flights between two airports on a date, by IATA code.",
    "parameters": {
        "type": "object",
        "properties": {
            "origin": {"type": "string", "description": "origin airport IATA, e.g. BOS"},
            "destination": {"type": "string", "description": "destination airport IATA, e.g. LHR"},
            "departureDate": {"type": "string", "description": "departure date YYYY-MM-DD, e.g. 2026-07-12"},
        },
        "required": ["origin", "destination"],
    },
}


async def run_tool(name, args):
    return await search_flights(args.get("origin", ""), args.get("destination", ""),
                                args.get("departureDate", ""))


@app.get("/.well-known/agent-card.json")
def agent_card():
    return AGENT_CARD


@app.get("/health")
def health():
    return {"ok": True}


async def _build_task(text: str) -> dict:
    """Run the loop and wrap the result as an A2A Task. Shared by BOTH transports —
    the Message in and the Task out are identical; only the envelope differs."""
    final_text, data, trace = await llm.run(SYSTEM, text, TOOL, run_tool)
    parts = [{"text": final_text}]
    if data and data.get("ok"):
        parts.append({"data": data})
    return {
        "id": "task-" + uuid.uuid4().hex[:6],
        "status": {"state": "completed"},
        "artifacts": [{"artifactId": "art-" + uuid.uuid4().hex[:6], "name": "flights", "parts": parts}],
        "metadata": {"provider": llm.provider_label(), "trace": trace + [{"phase": "A2A", "kind": "response",
                     "label": "A2A response", "detail": "Task(completed) + Artifact"}]},
    }


@app.post("/")
async def rpc(body: dict):
    """JSON-RPC 2.0 binding: {jsonrpc, id, method, params} -> {jsonrpc, id, result: Task}."""
    rpc_id = body.get("id")
    if body.get("method") != "message/send":
        return {"jsonrpc": "2.0", "id": rpc_id,
                "error": {"code": -32601, "message": "method not found"}}
    text = body["params"]["message"]["parts"][0].get("text", "")
    task = await _build_task(text)
    return {"jsonrpc": "2.0", "id": rpc_id, "result": task}


@app.post("/v1/message:send")
async def rest_send(body: dict):
    """HTTP+JSON (REST) binding: POST /v1/message:send with the Message as the body,
    returns the Task object directly (no JSON-RPC envelope)."""
    text = (body.get("message", {}).get("parts", [{}])[0]).get("text", "")
    return await _build_task(text)
