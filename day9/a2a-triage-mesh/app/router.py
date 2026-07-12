"""
router.py — the orchestrator agent. It owns NO clinical knowledge and has NO
hardcoded list of wards. It learns the whole capability graph by DISCOVERING the
specialist agents — reading each one's Agent Card — and then routes one case to
the matched specialist over A2A.

One streamed pass, in three acts:

  DISCOVER  GET every peer's /.well-known/agent-card.json. Each card declares
            what (system, acuity) tuples it claims; those become graph edges.
            A peer that's down simply contributes no edges — discovery is the
            single source of truth for what the mesh can do right now.

  TRIAGE    SENSE (the extraction contract + the patient's words) → REASON
            (extract_intent → a structured intent) → ACT (look the intent up in
            the discovered graph → a specialist URL, or refuse).

  DISPATCH  POST message/stream to that specialist's A2A url; stream its scoped
            tool loop and its disposition artifact back to the UI.
"""
import time

import httpx

import a2a
from config import cfg

ROUTER_SYSTEM = (
    "You are a hospital triage router. Read the patient's words and return ONLY a "
    "JSON object (no prose, no markdown) with keys:\n"
    '  "system": one of '
    '["cardiac","respiratory","neuro","dermatology","gastro","ortho","general"]\n'
    '  "acuity": "HIGH" or "LOW"\n'
    '  "signs": array of up to 4 short snake_case symptom tags\n'
    "If the message is not a clinical presentation, use "
    '{"system":"general","acuity":"LOW"}. Extract intent only; never route, never '
    "name a ward, never give medical advice."
)


async def _discover(client, peers):
    """GET each peer's card. Returns (agents, rows, events)."""
    events, agents, rows = [], [], []
    events.append({"phase": "discover_start", "peers": peers})
async def _discover(client, peers, down=None):
    """GET each peer's card. `down` is a set of role names to treat as unreachable
    (the UI's kill-a-container switch — same effect as `docker compose stop`)."""
    down = down or set()
    _PORT_ROLE = {"8101": "cardiology", "8102": "pulmonology",
                  "8103": "dermatology", "8104": "neurology"}
    events, agents, rows = [], [], []
    events.append({"phase": "discover_start", "peers": peers})
    for url in peers:
        card_url = f"{url.rstrip('/')}/.well-known/agent-card.json"
        role_guess = next((r for r in ("cardiology", "pulmonology", "dermatology", "neurology")
                           if r in url), None)
        if role_guess is None:
            port = url.rstrip("/").rsplit(":", 1)[-1].split("/")[0]
            role_guess = _PORT_ROLE.get(port, url)
        events.append({"phase": "probe", "url": url, "card_url": card_url})
        is_down = (role_guess in down) or (url in down) or any(d and d in url for d in down)
        if is_down:
            events.append({"phase": "down", "url": url, "role": role_guess,
                           "message": "ConnectionError (container stopped)"})
            continue
        try:
            r = await client.get(card_url)
            r.raise_for_status()
            card = r.json()
        except Exception as e:                               # noqa: BLE001
            events.append({"phase": "down", "url": url, "role": role_guess,
                           "message": f"{type(e).__name__}"})
            continue
        claims = card.get("x-triage-claims", [])
        streaming = bool((card.get("capabilities") or {}).get("streaming", False))
        agent = {"name": card.get("name", url), "ward": card.get("x-ward", ""),
                 "url": card.get("url", f"{url}/a2a"), "base": url,
                 "tools": card.get("x-tools", []), "claims": claims, "streaming": streaming,
                 "skill": (card.get("skills") or [{}])[0]}
        agents.append(agent)
        for sysname, acuity in claims:
            rows.append({"system": sysname, "acuity": acuity, "agent": agent["name"],
                         "ward": agent["ward"], "url": agent["url"], "streaming": streaming})
        events.append({"phase": "card", "url": url, "card": card,
                       "name": agent["name"], "ward": agent["ward"], "streaming": streaming,
                       "tools": agent["tools"], "claims": claims, "skill": agent["skill"]})
    events.append({"phase": "graph", "rows": rows, "agents": agents})
    return agents, rows, events


def _lookup(rows, system, acuity):
    for r in rows:
        if r["system"] == system and r["acuity"] == acuity:
            return r
    return None


async def run(presentation: str, engine: str, peers=None,
              transport: str = "auto", down=None, fail: bool = False):
    """Async generator of UI event dicts: discover → triage → dispatch.

    transport: "auto" (obey the matched agent's card capability), "stream", or "send".
    down:      optional set of role names to treat as unreachable (for tests; the
               live demo takes agents down by stopping the container in Docker).
    fail:      inject a tool failure in the dispatched specialist (→ failed task).
    """
    peers = cfg.PEERS if peers is None else peers
    down = set(down or [])
    from engine import get_provider
    try:
        provider = get_provider(engine)
    except Exception as e:                                   # noqa: BLE001
        yield {"phase": "error", "where": "engine", "message": str(e)}
        return

    async with httpx.AsyncClient(timeout=cfg.HTTP_TIMEOUT) as client:
        # ── DISCOVER ──
        agents, rows, dev = await _discover(client, peers, down=down)
        for ev in dev:
            yield ev

        # ── TRIAGE ──
        yield {"phase": "case", "presentation": presentation, "engine": engine}
        yield {"phase": "sense", "system_prompt": ROUTER_SYSTEM, "user": presentation}
        t0 = time.time()
        try:
            intent = provider.extract_intent(ROUTER_SYSTEM, presentation)
        except Exception as e:                               # noqa: BLE001
            yield {"phase": "error", "where": "reason", "message": f"{type(e).__name__}: {e}"}
            return
        ms = int((time.time() - t0) * 1000)
        system = str(intent.get("system", "general")).lower()
        acuity = str(intent.get("acuity", "LOW")).upper()
        signs = [str(s) for s in intent.get("signs", [])][:4]
        intent = {"system": system, "acuity": acuity, "signs": signs}
        yield {"phase": "reason", "intent": intent, "latency_ms": ms}

        match = _lookup(rows, system, acuity)
        yield {"phase": "act", "intent": intent,
               "agent": match["agent"] if match else None,
               "url": match["url"] if match else None,
               "ward": match["ward"] if match else None}

        if not match:
            yield {"phase": "reject", "intent": intent,
                   "reason": f"no discovered agent claims ({system}, {acuity})"}
            yield {"phase": "done"}
            return

        # ── TRANSPORT DECISION (capability-driven) ──
        card_streaming = bool(match.get("streaming", False))
        if transport == "stream":
            use_stream, why = True, "forced by override"
        elif transport == "send":
            use_stream, why = False, "forced by override"
        else:  # auto — obey the card
            use_stream = card_streaming
            why = ("card advertises streaming" if card_streaming
                   else "card advertises streaming:false → fall back to send")
        method = "message/stream" if use_stream else "message/send"
        yield {"phase": "transport", "agent": match["agent"], "method": method,
               "card_streaming": card_streaming, "mode": transport, "why": why}

        # ── DISPATCH over A2A ──
        yield {"phase": "dispatch", "agent": match["agent"], "url": match["url"],
               "ward": match["ward"], "method": method}
        meta = {"engine": engine}
        if fail:
            meta["fail"] = True
        rpc = {"jsonrpc": "2.0", "id": 1, "method": method,
               "params": {"message": a2a.user_message(presentation), "metadata": meta}}
        yield {"phase": "send", "url": match["url"], "payload": rpc, "method": method}

        disposition, first_ms, t_send = None, None, time.time()
        import json as _json

        if not use_stream:
            # ── message/send: ONE request, ONE response. The client is blind
            #    until the whole Task comes back (no intermediate frames). ──
            yield {"phase": "wait", "agent": match["agent"]}
            try:
                resp = await client.post(match["url"], json=rpc)
                resp.raise_for_status()
                result = (resp.json() or {}).get("result") or {}
            except Exception as e:                           # noqa: BLE001
                yield {"phase": "error", "where": "dispatch", "message": f"{type(e).__name__}: {e}"}
                return
            total_ms = int((time.time() - t_send) * 1000)
            first_ms = total_ms                              # first byte == last byte
            state = (result.get("status") or {}).get("state", "completed")
            arts = result.get("artifacts") or []
            art = arts[0] if arts else a2a.text_artifact("(no artifact)")
            disposition = a2a.first_text(art)
            yield {"phase": "task", "agent": match["agent"], "state": state,
                   "final": True, "transport": "send"}
            yield {"phase": "agent_artifact", "agent": match["agent"],
                   "artifact": art, "text": disposition, "transport": "send", "state": state}
            yield {"phase": "metrics", "transport": "send", "first_ms": first_ms,
                   "total_ms": total_ms, "frames": 1}
            yield {"phase": "return", "agent": match["agent"], "ward": match["ward"],
                   "text": disposition, "state": state}
            yield {"phase": "done"}
            return

        # ── message/stream: ONE request, MANY frames back ──
        frames = 0
        try:
            async with client.stream("POST", match["url"], json=rpc) as resp:
                resp.raise_for_status()
                async for line in resp.aiter_lines():
                    if not line or not line.startswith("data:"):
                        continue
                    if first_ms is None:
                        first_ms = int((time.time() - t_send) * 1000)
                    frames += 1
                    result = (_json.loads(line[5:].strip()).get("result") or {})
                    kind = result.get("kind")
                    if kind == "status-update":
                        yield {"phase": "task", "agent": match["agent"], "transport": "stream",
                               "state": result["status"]["state"], "final": result.get("final", False)}
                    elif kind == "agent-reason":
                        yield {"phase": "agent_reason", "agent": match["agent"],
                               "turn": result.get("turn"), "thought": result.get("thought"),
                               "tool": result.get("tool")}
                    elif kind == "agent-tool":
                        yield {"phase": "agent_tool", "agent": match["agent"],
                               "turn": result.get("turn"), "name": result.get("name"),
                               "ok": result.get("ok"), "result": result.get("result")}
                    elif kind == "artifact-update":
                        art = result.get("artifact") or {}
                        disposition = a2a.first_text(art)
                        yield {"phase": "agent_artifact", "agent": match["agent"],
                               "artifact": art, "text": disposition}
                    elif kind == "agent-error":
                        yield {"phase": "agent_error", "where": match["agent"],
                               "message": result.get("message")}
        except Exception as e:                               # noqa: BLE001
            yield {"phase": "error", "where": "dispatch", "message": f"{type(e).__name__}: {e}"}
            return

        total_ms = int((time.time() - t_send) * 1000)
        yield {"phase": "metrics", "transport": "stream", "first_ms": first_ms or 0,
               "total_ms": total_ms, "frames": frames}
        yield {"phase": "return", "agent": match["agent"], "ward": match["ward"], "text": disposition}
        yield {"phase": "done"}
