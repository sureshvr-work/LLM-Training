# ---------------------------------------------------------------------------
# a2a_client.py  ·  the orchestrator's two async styles: PUSH and PULL
# ---------------------------------------------------------------------------
# discover()          REST  ->  GET  {base}/.well-known/agent-card.json
#
# PUSH (Research Agent):
#   send_nonblocking()  RPC  ->  message/send, configuration.blocking=false +
#                                 pushNotificationConfig={url, token}. Returns
#                                 the sub-agent's immediate ack; the eventual
#                                 result arrives later as an inbound POST to
#                                 our own /webhook/tasks route. Nothing here
#                                 polls.
#
# PULL (Long-Task Agent):
#   send_async()   RPC  ->  message/send, configuration.blocking=false, NO
#                            push config. Same instant ack, but this time
#                            it's on US to check back in.
#   get_task()     RPC  ->  tasks/get {id} — a stateless snapshot, call it as
#                            often as you like.
#   cancel_task()  RPC  ->  tasks/cancel {id} — ask the agent to stop.
# ---------------------------------------------------------------------------

import uuid
import httpx


async def discover(base: str, trace: list) -> tuple:
    url = base.rstrip("/") + "/.well-known/agent-card.json"
    async with httpx.AsyncClient(timeout=8.0) as client:
        resp = await client.get(url)
    card = resp.json()
    trace.append({
        "phase": "DISCOVER", "kind": "discover", "binding": "REST",
        "label": "DISCOVER · fetched Agent Card",
        "detail": f"GET {url} · pushNotifications="
                  f"{(card.get('capabilities') or {}).get('pushNotifications', False)}",
    })
    return card, trace[-1]


async def send_nonblocking(base: str, text: str, webhook_url: str, token: str, trace: list) -> dict:
    """message/send with blocking=false + a pushNotificationConfig. Returns
    the sub-agent's immediate ack (Task in state 'submitted') — this call
    itself completes in well under a second even though the real work will
    take much longer."""
    message = {"role": "user", "messageId": "msg-" + uuid.uuid4().hex[:6], "parts": [{"text": text}]}
    request = {
        "jsonrpc": "2.0", "id": uuid.uuid4().hex[:8], "method": "message/send",
        "params": {
            "message": message,
            "configuration": {
                "blocking": False,
                "pushNotificationConfig": {"url": webhook_url, "token": token},
            },
        },
    }
    trace.append({
        "kind": "call", "label": f"message/send (non-blocking) → {base}",
        "detail": f'JSON-RPC 2.0 · configuration.blocking=false · '
                  f'pushNotificationConfig.url={webhook_url} · "{text}"',
    })
    async with httpx.AsyncClient(timeout=10.0) as client:
        resp = await client.post(base.rstrip("/") + "/", json=request)
    body = resp.json()
    if "error" in body:
        return {"error": body["error"]}
    return body.get("result", {})


async def send_async(base: str, text: str, trace: list) -> dict:
    """message/send with blocking=false and NO pushNotificationConfig — the
    PULL model. The sub-agent still ACKs instantly and works in the
    background, but delivery is now our responsibility via tasks/get."""
    message = {"role": "user", "messageId": "msg-" + uuid.uuid4().hex[:6], "parts": [{"text": text}]}
    request = {
        "jsonrpc": "2.0", "id": uuid.uuid4().hex[:8], "method": "message/send",
        "params": {"message": message, "configuration": {"blocking": False}},
    }
    trace.append({
        "kind": "call", "label": f"message/send (non-blocking, no push) → {base}",
        "detail": f'JSON-RPC 2.0 · configuration.blocking=false · "{text}"',
    })
    async with httpx.AsyncClient(timeout=10.0) as client:
        resp = await client.post(base.rstrip("/") + "/", json=request)
    body = resp.json()
    if "error" in body:
        return {"error": body["error"]}
    return body.get("result", {})


async def get_task(base: str, task_id: str, trace: list) -> dict:
    """tasks/get — a single stateless poll. Every call is a fresh request;
    the sub-agent doesn't remember that we've asked before."""
    request = {"jsonrpc": "2.0", "id": uuid.uuid4().hex[:8], "method": "tasks/get", "params": {"id": task_id}}
    trace.append({"kind": "call", "label": f"tasks/get → {base}", "detail": f"JSON-RPC 2.0 · id={task_id}"})
    async with httpx.AsyncClient(timeout=10.0) as client:
        resp = await client.post(base.rstrip("/") + "/", json=request)
    body = resp.json()
    if "error" in body:
        return {"error": body["error"]}
    return body.get("result", {})


async def cancel_task(base: str, task_id: str, trace: list) -> dict:
    """tasks/cancel — ask the sub-agent to stop. Fails with -32002 if the
    task is already in a terminal state (completed/canceled/failed)."""
    request = {"jsonrpc": "2.0", "id": uuid.uuid4().hex[:8], "method": "tasks/cancel", "params": {"id": task_id}}
    trace.append({"kind": "call", "label": f"tasks/cancel → {base}", "detail": f"JSON-RPC 2.0 · id={task_id}"})
    async with httpx.AsyncClient(timeout=10.0) as client:
        resp = await client.post(base.rstrip("/") + "/", json=request)
    body = resp.json()
    if "error" in body:
        return {"error": body["error"]}
    return body.get("result", {})
