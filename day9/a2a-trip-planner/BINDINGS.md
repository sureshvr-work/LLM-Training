# A2A transport bindings — REST vs JSON-RPC vs gRPC

Same protocol, same objects (AgentCard, Message, Task, Artifact). Only the
**envelope on the wire** changes. Discovery is always a plain `GET`; the
*message* is what differs. This demo uses **REST for discovery** and
**JSON-RPC 2.0 for messaging** (`trip-planner/a2a_client.py`).

The Message object below is identical in all three — that's the whole point.

---

## 1. REST (what the discovery call uses)

Plain HTTP verbs and paths. Discovery:

```
GET /.well-known/agent-card.json            ->  the Agent Card
```

A REST message send looks like:

```http
POST /v1/message:send
Content-Type: application/json

{ "message": { "role": "user", "parts": [ { "text": "Weather in Paris tomorrow?" } ] } }
```

```python
resp = await client.post(base + "/v1/message:send", json={"message": {...}})
task = resp.json()                     # the Task, directly
```

---

## 2. JSON-RPC 2.0 (what this demo's messaging uses)

One endpoint (`POST /`), and the *method* is named in the body. The A2A payload
sits in `params`; the result comes back in `result`.

```http
POST /
Content-Type: application/json

{ "jsonrpc": "2.0", "id": "42", "method": "message/send",
  "params": { "message": { "role": "user", "parts": [ { "text": "Weather in Paris tomorrow?" } ] } } }
```

Response:

```json
{ "jsonrpc": "2.0", "id": "42", "result": { "id": "task-…", "status": {"state":"completed"}, "artifacts": [ … ] } }
```

```python
req = {"jsonrpc": "2.0", "id": "42", "method": "message/send",
       "params": {"message": {"role": "user", "parts": [{"text": "..."}]}}}
resp = await client.post(base + "/", json=req)
task = resp.json()["result"]           # note: unwrap "result"
```

Compare with the REST snippet you already had — the *only* differences are the
`jsonrpc/id/method` wrapper and unwrapping `result`. The Message is byte-for-byte
the same.

---

## 3. gRPC (binary, HTTP/2, needs a `.proto`)

gRPC isn't hand-written JSON over HTTP — it's a typed service defined in a
`.proto`, compiled to client/server stubs. A2A defines it roughly like:

```protobuf
service A2AService {
  rpc SendMessage (SendMessageRequest) returns (Task);
  rpc GetTask     (GetTaskRequest)     returns (Task);
  rpc GetAgentCard(GetAgentCardRequest) returns (AgentCard);
}

message Message { string role = 1; repeated Part parts = 2; string message_id = 3; }
message Part    { oneof content { string text = 1; bytes raw = 2; string uri = 3; } }
```

You generate stubs and *call a method* instead of POSTing JSON:

```python
# after: python -m grpc_tools.protoc ... a2a.proto   (generates a2a_pb2, a2a_pb2_grpc)
import grpc, a2a_pb2, a2a_pb2_grpc

channel = grpc.aio.insecure_channel("weather-agent:8001")
stub = a2a_pb2_grpc.A2AServiceStub(channel)

msg = a2a_pb2.Message(role="user", parts=[a2a_pb2.Part(text="Weather in Paris tomorrow?")])
task = await stub.SendMessage(a2a_pb2.SendMessageRequest(message=msg))   # returns a typed Task
```

Why it's not in this demo: it needs the `.proto`, a codegen step, `grpcio`, and
HTTP/2 plumbing — great for high-throughput production, heavy for a teaching
walkthrough. The lesson to say out loud: **the Message and Task are the same
objects; gRPC just carries them as typed binary over HTTP/2 instead of JSON over
HTTP/1.** Read the object once, understand it on every binding.

---

### Which to reach for
- **REST** — easiest to read/debug; good default for humans and demos.
- **JSON-RPC** — A2A's default; one endpoint, method-in-body; trivial to route.
- **gRPC** — typed, compact, fast; for service-to-service at scale.

An agent's card announces its choice in `preferredTransport`, and may list more
in `additionalInterfaces`. The caller reads that and speaks whichever it likes.

## Live in this demo: two agents, two transports

The two agents now run **different bindings on purpose**, so you can watch the same
Message/Task objects cross different envelopes:

- **Weather Agent** — `preferredTransport: JSONRPC`. Messaged at `POST /` with a
  JSON-RPC 2.0 envelope; the Task comes back in `result`.
- **Flights Agent** — `preferredTransport: HTTP+JSON`. Messaged at `POST /v1/message:send`
  with the Message as the raw body; the Task comes back **directly** (no envelope).

Both also declare the other binding in `additionalInterfaces`, and the Flights Agent
literally implements both endpoints — they share one `_build_task()`, proving the point:
the transport is just the envelope; the Message in and Task out are identical.

The client (`a2a_client.send`) reads each card's `preferredTransport` after discovery and
picks the binding — exactly what a real A2A client does.

## Streaming: message/stream (the Flight Status agent)

The Flight Status agent (port 8004) advertises `capabilities.streaming: true` and answers
`message/stream` instead of `message/send`. Same Message in — but the server holds the HTTP
connection open and pushes **Server-Sent Events**, each `data:` line a JSON-RPC response whose
result is a `TaskStatusUpdateEvent`. The task stays `working` while the flight moves
(boarding → departed → en route → landing) and closes on the terminal event
(`state: completed`, `final: true`).

Two layers worth calling out in class: the **task lifecycle** (`submitted → working → completed`)
is the A2A envelope; the **flight phase** (Scheduled/Boarding/Landed…) rides inside the status
*message*. One is the protocol, the other is the domain.

The orchestrator consumes that SSE with `a2a_client.stream()` and **relays** it to the browser
over its own `POST /status/stream` — the client reads the card, sees streaming is supported, and
opens the stream. That passthrough is the whole client half of the streaming lesson.

## Multi-turn: input-required (the Booking agent)

The Booking agent (port 8005) shows a Task that lives across turns. Turn 1 has passenger + flight
but not seat / date-of-birth, so the agent pauses the task at `state: input-required` and asks. The
client answers with another `message/send` carrying the **same taskId** (and contextId); the agent
loads the stored fields, merges the answer, and finishes at `completed` with a PNR artifact.

The lesson: the `taskId` is the thread — it's how a stateful conversation stays one task across
several messages. `contextId` groups the whole exchange; `taskId` is this one request within it.
The agent holds partial state server-side keyed by `taskId`; a terminal state frees it.

The orchestrator drives it as two calls (`POST /book/start` then `POST /book/reply`), and
`a2a_client.send(..., task_id=, context_id=)` is what threads the reply onto the in-flight task.
