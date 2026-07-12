# A2A Fire-and-Forget — PUSH (webhook) and PULL (poll/cancel), side by side

An **Orchestrator** fires a request at a slow sub-agent and does not wait for
it. Two sub-agents show the two ways A2A lets you find out what happened
later: the **Research Agent** calls the orchestrator back over a **webhook**
(push); the **Long-Task Agent** does nothing on its own — the orchestrator has
to **poll** it, and can **cancel** it mid-flight (pull).

```
research-agent  ──(webhook POST when done)──►  orchestrator  ──►  ui  (http://localhost:8090)
     ▲                                               │
     └────── message/send, blocking=false ───────────┘   PUSH

long-task-agent  ◄──(tasks/get · tasks/cancel)──  orchestrator                PULL
     ▲                                               │
     └────── message/send, blocking=false ───────────┘
               (both return in ms, work continues after)
```

## Is this possible from a webhook? Yes — it's a named part of A2A

A2A calls this **push notifications**. It's not a hack bolted onto the
protocol — `PushNotificationConfig` and `MessageSendConfiguration.blocking`
are first-class fields in the spec, built for exactly this: work that's too
slow for a client to sit on an open connection for (an SSE stream still needs
both sides alive and connected; a webhook doesn't).

The handshake, concretely (see `research-agent/app.py` and
`orchestrator/app.py` — every piece below is a real field, not a shortcut):

1. **Orchestrator sends non-blocking, with a callback.** Its `message/send`
   sets `configuration.blocking = false` and
   `configuration.pushNotificationConfig = { url, token }` — `url` is the
   orchestrator's own `/webhook/tasks` route, `token` is a secret minted
   *per task* so the callback can be verified later.
2. **Sub-agent ACKs immediately.** It returns a `Task` in state `"submitted"`
   — no artifact yet — the instant it has accepted the work. That HTTP
   request/response is now **over**. `orchestrator/app.py`'s `/research/start`
   returns to the UI in milliseconds, not seconds.
3. **Sub-agent keeps working, disconnected.** `research-agent/app.py` spins
   the actual work off with `asyncio.create_task(...)`, fully decoupled from
   the request that triggered it. It could take 10 seconds or 10 minutes —
   nothing is holding a socket open for it.
4. **Sub-agent calls back.** When done, it `POST`s the finished `Task`
   (`status.completed` + `Artifact`) to the `url` from step 1, with the
   `token` from step 1 echoed back in an `X-A2A-Notification-Token` header.
   `orchestrator/app.py`'s `/webhook/tasks` checks that token against what it
   minted before trusting the payload — **always verify inbound webhooks**,
   since the URL is, in principle, POST-able by anyone who finds it.
5. **Delivery is retried, not guaranteed instant.** `_notify_webhook()` in
   `research-agent/app.py` retries with backoff if the callback fails — a
   webhook is at-least-once delivery, unlike a stream where "the connection
   is still open" *is* the delivery guarantee.

### How this differs from what's already in `a2a-trip-planner`

| | held connection? | client polls? | survives a restart? | good for |
|---|---|---|---|---|
| `message/send` (blocking) | yes, until done | no | no | fast calls (weather, flights) |
| `message/stream` (SSE) | yes, whole time | no (server pushes) | no — reconnect loses position | live progress (flight status) |
| **push notification** (Research Agent) | **no** | **no** | **yes** — webhook can arrive after either side restarts | slow/long-running jobs, work that outlives the request |
| **poll + cancel** (Long-Task Agent) | no | **yes**, on a timer | yes — `tasks/get` works any time | jobs you may want to abort, or where you can't run a public webhook endpoint |

The SSE relay you'll see in the UI (`GET /research/{id}/stream`) is **not**
part of the A2A hop — it's just how the orchestrator nicely pushes "the
webhook just arrived" to the browser. The real fire-and-forget leg
(orchestrator → research-agent → webhook) has zero open connections at any
point after step 2. The README calls this out again inline in
`orchestrator/app.py` so it isn't confused with streaming.

## The other half: `tasks/get` and `tasks/cancel` (pull model)

Push notifications need a public webhook endpoint, which isn't always
possible (locked-down networks, no ingress, client behind NAT). A2A's answer
for that case is the same non-blocking `message/send`, but the client is
responsible for checking in — and can stop the job outright. Both are
standard JSON-RPC methods, not something this repo invented:

- **`tasks/get`** — `{"method": "tasks/get", "params": {"id": "<taskId>"}}`.
  A stateless snapshot of the `Task` right now. Call it once, call it every
  2 seconds, doesn't matter — the agent doesn't track who's asking or how
  often (see `long-task-agent/app.py`'s `tasks/get` branch).
- **`tasks/cancel`** — `{"method": "tasks/cancel", "params": {"id": "<taskId>"}}`.
  Asks the agent to stop. `long-task-agent/app.py` cancels the underlying
  `asyncio.Task` and the state becomes `"canceled"` — a *terminal* state,
  same family as `"completed"`. Calling `tasks/cancel` again (or on a task
  that already finished) gets A2A's `TaskNotCancelableError` (code `-32002`).

`orchestrator/app.py`'s `/long/start`, `/long/{id}` (poll), and
`/long/{id}/cancel` routes wrap these three calls; the UI's second panel
starts a job, polls it every 2s, and lets you cancel mid-flight.

## Run

```bash
cp .env.example .env         # add OPENAI_API_KEY (or ANTHROPIC)
docker compose up -d --build
# open http://localhost:8090
```

Curl the handshake directly, without the UI:

```bash
# 1) discover — does it advertise pushNotifications?
curl http://localhost:8101/.well-known/agent-card.json | grep -A2 capabilities

# 2) start a run through the orchestrator (this returns almost instantly)
curl -s -X POST http://localhost:8103/research/start \
  -H 'Content-Type: application/json' \
  -d '{"destination":"Lisbon","days":3}'
# => {"taskId":"task-xxxxxx","state":"submitted","ackMs":180,...}

# 3) poll the orchestrator's own (cheap, local) view while you wait
curl http://localhost:8103/research/task-xxxxxx

# 4) — or watch the research agent's own state flip submitted -> completed —
curl http://localhost:8101/debug/tasks/task-xxxxxx
```

And the pull side — poll it, then cancel it before it finishes:

```bash
# 1) start a 30s job through the orchestrator
curl -s -X POST http://localhost:8103/long/start -H 'Content-Type: application/json' -d '{"seconds":30}'
# => {"taskId":"task-yyyyyy","state":"submitted",...}

# 2) poll it — a fresh tasks/get every time, nothing cached
curl http://localhost:8103/long/task-yyyyyy
# => {"taskId":"task-yyyyyy","state":"working",...}

# 3) change your mind
curl -X POST http://localhost:8103/long/task-yyyyyy/cancel
# => {"taskId":"task-yyyyyy","state":"canceled",...}

# 4) cancel it again — the spec's TaskNotCancelableError
curl -i -X POST http://localhost:8103/long/task-yyyyyy/cancel
# => HTTP 409, {"error":{"code":-32002,"message":"task cannot be canceled in state canceled"}}
```

## What to show in class

1. **The gap.** Send a request and watch the UI timeline: the orchestrator's
   HTTP call to the sub-agent finishes in milliseconds ("ack received"), then
   there's a visible dead period — nothing polling, nothing streaming — before
   a *second, independent* inbound request (the webhook) delivers the result.
2. **Two different HTTP requests, two different directions.** Request 1:
   orchestrator → research-agent (`message/send`). Request 2, later, and
   backwards: research-agent → orchestrator (`POST /webhook/tasks`). A2A
   doesn't require the same connection, or even the same machine, to still be
   around for the second one.
3. **Verify the webhook.** Show `orchestrator/app.py`'s token check
   (`token != SECRETS[tid]`) rejecting a forged callback with 401 — an
   endpoint the internet can `POST` to needs that.
4. **Bump `RESEARCH_SECONDS`** in `.env` to make the gap dramatic (a minute or
   more), refresh the browser mid-run, and reload `/research/{taskId}` — the
   result still shows up, because delivery didn't depend on that browser tab
   or that orchestrator process staying up.
5. **Push vs. pull, side by side.** Run both panels. Both start the same way
   (non-blocking `message/send`, instant ack) — the only difference is what
   happens *after*: one waits for an inbound call, the other goes and asks.
6. **Cancel mid-flight.** Start a 30s Long-Task job, cancel it at ~5s, and
   show `long-task-agent`'s `_run()` catching `asyncio.CancelledError` and
   flipping state to `"canceled"` — then try cancelling it again and get
   `TaskNotCancelableError`, because it's already in a terminal state.

## Files

```
research-agent/   app.py · llm.py            (accepts non-blocking sends, POSTs results to a webhook — PUSH)
long-task-agent/  app.py                     (accepts non-blocking sends, answers tasks/get + tasks/cancel — PULL)
orchestrator/     app.py · a2a_client.py     (sends non-blocking either way; webhook receiver + SSE relay; poll/cancel routes)
ui/               app.py · index.html        (two panels: fire button + live timeline, and start/poll/cancel)
```

`llm.py` is copied verbatim from `a2a-trip-planner/weather-agent` — same real
SENSE → REASON → ACT loop, no mock, provider-switchable
(`LLM_PROVIDER=openai|anthropic`).

## Config (.env)

```
LLM_PROVIDER=openai            # openai | anthropic
OPENAI_API_KEY=...             OPENAI_MODEL=gpt-4o-mini
ANTHROPIC_API_KEY=...          ANTHROPIC_MODEL=claude-3-5-sonnet-latest
RESEARCH_SECONDS=12            # simulated background work duration
```
