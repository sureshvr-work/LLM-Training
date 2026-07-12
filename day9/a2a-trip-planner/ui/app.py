# ---------------------------------------------------------------------------
# UI backend  ·  serves the page and forwards trip requests to the Trip Planner
# ---------------------------------------------------------------------------

import os
from pathlib import Path

import httpx
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse

app = FastAPI(title="Trip Planner UI")

PLANNER_URL = os.getenv("PLANNER_URL", "http://trip-planner:8003")


@app.get("/", response_class=HTMLResponse)
def index():
    return Path(__file__).with_name("index.html").read_text(encoding="utf-8")


async def _forward(path: str, body: dict):
    try:
        async with httpx.AsyncClient(timeout=90.0) as client:
            resp = await client.post(PLANNER_URL.rstrip("/") + path, json=body)
        return resp.json()
    except httpx.HTTPError as exc:
        return JSONResponse({"error": f"could not reach the Trip Planner: {exc}"}, status_code=502)


@app.post("/api/plan")
async def plan(body: dict):
    return await _forward("/plan", body)


@app.post("/api/discover")
async def discover(body: dict):
    return await _forward("/discover", body)


@app.post("/api/compose")
async def compose(body: dict):
    return await _forward("/compose", body)


@app.post("/api/send")
async def send(body: dict):
    return await _forward("/send", body)


@app.get("/api/status-stream")
async def status_stream(flight: str = ""):
    """Proxy the Trip Planner's SSE straight to the browser's EventSource (GET)."""
    async def gen():
        async with httpx.AsyncClient(timeout=None) as client:
            async with client.stream("POST", PLANNER_URL.rstrip("/") + "/status/stream",
                                     json={"flight": flight}) as resp:
                async for chunk in resp.aiter_raw():
                    yield chunk
    return StreamingResponse(gen(), media_type="text/event-stream")


@app.post("/api/book/start")
async def book_start(body: dict):
    return await _forward("/book/start", body)


@app.post("/api/book/reply")
async def book_reply(body: dict):
    return await _forward("/book/reply", body)
