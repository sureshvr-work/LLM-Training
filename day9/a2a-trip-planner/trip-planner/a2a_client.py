# ---------------------------------------------------------------------------
# a2a_client.py  ·  how one agent talks to another over A2A
# ---------------------------------------------------------------------------
# Two calls, two bindings — this is the whole client side of A2A:
#
#   discover(base)      REST  ->  GET  {base}/.well-known/agent-card.json
#   send(base, text)    RPC   ->  POST {base}  with a JSON-RPC 2.0 envelope
#
# Discovery is always a plain GET. Messaging here uses JSON-RPC 2.0 (A2A's
# default binding). The same message over REST or gRPC would carry the exact
# same Message object — only the envelope changes. See BINDINGS.md.
# ---------------------------------------------------------------------------

import json
import uuid
import httpx


async def discover(base: str, trace: list) -> tuple:
    """A2A discovery — REST GET of the well-known Agent Card. Returns (card, step)."""
    url = base.rstrip("/") + "/.well-known/agent-card.json"
    async with httpx.AsyncClient(timeout=8.0) as client:
        resp = await client.get(url)
    card = resp.json()
    step = {
        "phase": "DISCOVER",
        "kind": "discover",
        "binding": "REST",
        "label": "DISCOVER · fetched Agent Card",
        "detail": f"GET {url}  ·  card says transport={card.get('preferredTransport')}",
    }
    trace.append(step)
    return card, step


async def send(base: str, text: str, trace: list, transport: str = "JSONRPC",
               task_id: str = None, context_id: str = None) -> dict:
    """A2A message send. Picks the binding from the card's transport. The Message
    object is identical either way — only the envelope + endpoint change.
    Pass task_id/context_id to CONTINUE an existing task (multi-turn)."""
    # The A2A Message — the SAME object crosses whichever binding we choose.
    message = {
        "role": "user",
        "messageId": "msg-" + uuid.uuid4().hex[:6],
        "parts": [{"text": text}],
    }
    if task_id:                      # threading the reply onto an in-flight task
        message["taskId"] = task_id
    if context_id:
        message["contextId"] = context_id

    if transport == "HTTP+JSON":
        # REST binding: POST {base}/v1/message:send with the message as the body;
        # the response IS the Task object (no JSON-RPC wrapper).
        url = base.rstrip("/") + "/v1/message:send"
        trace.append({"kind": "call", "label": f"message:send → {base}",
                      "detail": f'HTTP+JSON · POST /v1/message:send · "{text}"'})
        async with httpx.AsyncClient(timeout=45.0) as client:
            resp = await client.post(url, json={"message": message})
        body = resp.json()
        if isinstance(body, dict) and body.get("error"):
            return {"error": body["error"]}
        return body  # the Task, directly

    # JSON-RPC 2.0 binding (default): POST {base} with the {jsonrpc,id,method,params} envelope.
    url = base.rstrip("/") + "/"
    request = {
        "jsonrpc": "2.0",
        "id": uuid.uuid4().hex[:8],
        "method": "message/send",
        "params": {"message": message},
    }
    trace.append({"kind": "call", "label": f"message/send → {base}",
                  "detail": f'JSON-RPC 2.0 · method="message/send" · "{text}"'})
    async with httpx.AsyncClient(timeout=45.0) as client:
        resp = await client.post(url, json=request)
    body = resp.json()
    if "error" in body:
        return {"error": body["error"]}
    return body.get("result", {})


async def stream(base: str, text: str, trace: list, transport: str = "JSONRPC"):
    """A2A message/stream — open an SSE connection and yield each event's result.

    Same Message object as message/send; the only differences are the method name
    and that the server holds the connection open, pushing Server-Sent Events until
    a terminal event (final=true). This is the client half of the streaming lesson.
    """
    message = {"role": "user", "messageId": "msg-" + uuid.uuid4().hex[:6], "parts": [{"text": text}]}
    request = {"jsonrpc": "2.0", "id": uuid.uuid4().hex[:8],
               "method": "message/stream", "params": {"message": message}}
    trace.append({"kind": "call", "label": f"message/stream → {base}",
                  "detail": f'JSON-RPC 2.0 · method="message/stream" · SSE · "{text}"'})
    url = base.rstrip("/") + "/"
    async with httpx.AsyncClient(timeout=None) as client:
        async with client.stream("POST", url, json=request) as resp:
            async for line in resp.aiter_lines():
                line = line.strip()
                if line.startswith("data:"):
                    payload = json.loads(line[5:].strip())
                    yield payload.get("result", payload)
