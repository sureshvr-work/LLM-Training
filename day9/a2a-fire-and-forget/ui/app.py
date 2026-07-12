# ---------------------------------------------------------------------------
# UI backend  ·  serves the page and forwards requests to the Orchestrator
# ---------------------------------------------------------------------------

import os
from pathlib import Path

import httpx
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse

app = FastAPI(title="Fire-and-Forget UI")

ORCH_URL = os.getenv("ORCH_URL", "http://orchestrator:8103")


@app.get("/", response_class=HTMLResponse)
def index():
    return Path(__file__).with_name("index.html").read_text(encoding="utf-8")


@app.post("/api/research/start")
async def start(body: dict):
    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.post(ORCH_URL.rstrip("/") + "/research/start", json=body)
        return resp.json()
    except httpx.HTTPError as exc:
        return JSONResponse({"error": f"could not reach the orchestrator: {exc}"}, status_code=502)


@app.get("/api/research/{task_id}/stream")
async def stream(task_id: str):
    async def gen():
        async with httpx.AsyncClient(timeout=None) as client:
            async with client.stream("GET", ORCH_URL.rstrip("/") + f"/research/{task_id}/stream") as resp:
                async for chunk in resp.aiter_raw():
                    yield chunk
    return StreamingResponse(gen(), media_type="text/event-stream")


@app.post("/api/long/start")
async def long_start(body: dict):
    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.post(ORCH_URL.rstrip("/") + "/long/start", json=body)
        return resp.json()
    except httpx.HTTPError as exc:
        return JSONResponse({"error": f"could not reach the orchestrator: {exc}"}, status_code=502)


@app.get("/api/long/{task_id}")
async def long_poll(task_id: str):
    async with httpx.AsyncClient(timeout=15.0) as client:
        resp = await client.get(ORCH_URL.rstrip("/") + f"/long/{task_id}")
    return JSONResponse(resp.json(), status_code=resp.status_code)


@app.post("/api/long/{task_id}/cancel")
async def long_cancel(task_id: str):
    async with httpx.AsyncClient(timeout=15.0) as client:
        resp = await client.post(ORCH_URL.rstrip("/") + f"/long/{task_id}/cancel")
    return JSONResponse(resp.json(), status_code=resp.status_code)
