# ---------------------------------------------------------------------------
# Weather Agent  ·  a complete A2A agent in one file
# ---------------------------------------------------------------------------
# Two A2A surfaces:
#   GET  /.well-known/agent-card.json   -> discovery (REST)
#   POST /                              -> messages   (JSON-RPC 2.0, the A2A default)
#
# Inside message/send it runs the real LLM loop (llm.py) with its one real tool
# (weather_api.get_forecast) and returns a Task carrying a structured Artifact.
# ---------------------------------------------------------------------------

import uuid
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

import llm
from weather_api import get_forecast

app = FastAPI(title="Weather Agent")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

AGENT_CARD = {
    "protocolVersion": "1.0",
    "name": "Weather Agent",
    "description": "Answers weather questions for a city and day.",
    "url": "http://weather-agent:8001/",          # JSON-RPC endpoint
    "version": "1.0.0",
    "preferredTransport": "JSONRPC",
    "additionalInterfaces": [
        {"url": "http://weather-agent:8001/", "transport": "JSONRPC"},
    ],
    "capabilities": {"streaming": False},
    "skills": [{
        "id": "get-forecast",
        "name": "Get forecast",
        "description": "Return a short weather forecast for a place and time.",
        "tags": ["weather", "forecast"],
        "examples": ["What's the weather in Paris tomorrow?"],
    }],
}

SYSTEM = ("You are the Weather Agent. Call get_forecast to answer, then reply in one friendly sentence. "
          "Always pass a real CITY name. If the location is a state, region, or country (e.g. 'New Jersey'), "
          "use its largest or most representative city (e.g. 'Newark'). Fix obvious typos and missing spaces "
          "(e.g. 'newjersey' -> 'Newark', 'newyork' -> 'New York').")

# The tool as the model sees it (one neutral schema; llm.py adapts per provider).
TOOL = {
    "name": "get_forecast",
    "description": "Get the weather forecast for a city and day.",
    "parameters": {
        "type": "object",
        "properties": {
            "city": {"type": "string", "description": "city name, e.g. Paris"},
            "when": {"type": "string", "description": "'today' or 'tomorrow'"},
        },
        "required": ["city"],
    },
}


async def run_tool(name, args):
    # ACT: the agent (not the model) actually calls the real weather API.
    return await get_forecast(args.get("city", ""), args.get("when", "tomorrow"))


# ---- A2A discovery (REST) -------------------------------------------------
@app.get("/.well-known/agent-card.json")
def agent_card():
    return AGENT_CARD


@app.get("/health")
def health():
    return {"ok": True}


# ---- A2A messaging (JSON-RPC 2.0) -----------------------------------------
@app.post("/")
async def rpc(body: dict):
    rpc_id = body.get("id")
    if body.get("method") != "message/send":
        return {"jsonrpc": "2.0", "id": rpc_id,
                "error": {"code": -32601, "message": "method not found"}}

    text = body["params"]["message"]["parts"][0].get("text", "")
    final_text, data, trace = await llm.run(SYSTEM, text, TOOL, run_tool)

    # Rewrap into an A2A Task with a structured Artifact (text + data parts).
    parts = [{"text": final_text}]
    if data and data.get("ok"):
        parts.append({"data": data})       # the structured card the UI renders
    task = {
        "id": "task-" + uuid.uuid4().hex[:6],
        "status": {"state": "completed"},
        "artifacts": [{"artifactId": "art-" + uuid.uuid4().hex[:6],
                       "name": "forecast", "parts": parts}],
        "metadata": {"provider": llm.provider_label(), "trace": trace + [{"phase": "A2A", "kind": "response",
                     "label": "A2A response", "detail": "Task(completed) + Artifact"}]},
    }
    return {"jsonrpc": "2.0", "id": rpc_id, "result": task}
