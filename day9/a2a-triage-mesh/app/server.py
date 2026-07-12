"""
server.py — one image, any role. AGENT_ROLE decides what this process is.

  AGENT_ROLE=cardiology|pulmonology|dermatology|neurology   -> a SPECIALIST:
      GET  /.well-known/agent-card.json   its card (claims + skill + url)
      POST /a2a                            JSON-RPC: message/send, message/stream
      GET  /healthz

  AGENT_ROLE=router (default)                               -> the ROUTER + UI:
      POST /run        discover the mesh, triage one case, dispatch over A2A (SSE)
      GET  /agents     the live discovered graph (for first paint)
      GET  /healthz
      GET  /           the animated frontend

Run one process:   uvicorn server:app --port 8000
Run the mesh:       docker compose up --build   (5 services)
Default engine is `mock`, so the whole thing runs with NO API key.
"""
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel

import a2a
from config import cfg
from engine import get_provider

SPECIALIST_ROLES = ("cardiology", "pulmonology", "dermatology", "neurology")


def make_app(role: str = None, self_url: str = None, peers=None, serve_ui: bool = True,
             streaming: bool = None) -> FastAPI:
    role = (role or cfg.ROLE)
    self_url = cfg.SELF_URL if self_url is None else self_url
    peers = cfg.PEERS if peers is None else peers
    app = FastAPI(title=f"A2A triage mesh · {role}")

    # ── SPECIALIST ──
    if role in SPECIALIST_ROLES:
        import specialists

        @app.get("/.well-known/agent-card.json")
        def card(request: Request):
            base = self_url or str(request.base_url).rstrip("/")
            return JSONResponse(specialists.build_card(role, base, streaming=streaming))

        @app.post("/a2a")
        async def a2a_endpoint(request: Request):
            try:
                req = await request.json()
            except Exception:
                return JSONResponse(a2a.rpc_error(None, -32700, "parse error"))
            method = req.get("method")
            engine = ((req.get("params") or {}).get("metadata") or {}).get("engine", "mock")
            try:
                provider = get_provider(engine)
            except Exception as e:  # noqa: BLE001
                return JSONResponse(a2a.rpc_error(req.get("id"), -32602, str(e)))
            if method == "message/stream":
                return StreamingResponse(specialists.handle_stream(role, req, provider),
                                         media_type="text/event-stream")
            if method == "message/send":
                return JSONResponse(specialists.handle_send(role, req, provider))
            return JSONResponse(a2a.rpc_error(req.get("id"), -32601, f"method not found: {method}"))

        @app.get("/healthz")
        def healthz():
            return {"ok": True, "role": role}
        return app

    # ── ROUTER (+ UI) ──
    import router as router_mod

    class RunReq(BaseModel):
        presentation: str
        engine: str = "mock"
        transport: str = "auto"          # "auto" | "stream" | "send"
        down: list[str] = []             # role names to treat as stopped containers
        fail: bool = False               # inject a tool failure in the specialist

    @app.post("/run")
    def run(req: RunReq):
        async def gen():
            async for event in router_mod.run(req.presentation, req.engine, peers=peers,
                                              transport=req.transport, down=req.down, fail=req.fail):
                yield a2a.sse(event)
        return StreamingResponse(gen(), media_type="text/event-stream")

    @app.get("/agents")
    async def agents():
        import httpx
        async with httpx.AsyncClient(timeout=cfg.HTTP_TIMEOUT) as client:
            _agents, rows, _events = await router_mod._discover(client, peers)
        return {"agents": _agents, "rows": rows, "peers": peers}

    @app.get("/healthz")
    def healthz_r():
        return {"ok": True, "role": "router", "peers": peers}

    if serve_ui:
        from fastapi.staticfiles import StaticFiles
        app.mount("/", StaticFiles(directory="../frontend", html=True), name="frontend")
    return app


# Default app for uvicorn (role/peers from env).
app = make_app()
