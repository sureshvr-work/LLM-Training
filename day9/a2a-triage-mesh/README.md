# A2A triage mesh · discovery across containers

Five services, one network. A **Router** is handed a patient presentation but owns
no clinical knowledge and **no hardcoded list of wards**. It **discovers** four
specialist agents — each running in its **own container** — by reading their A2A
Agent Cards, assembles a capability graph from what they declare, and routes the
case to the matched specialist over real A2A.

```
                       ┌──────────────┐
        discover ┌────▶│ Cardiology   │  cardiac · HIGH/LOW
        (GET     │     ├──────────────┤
        card)    ├────▶│ Pulmonology  │  respiratory · HIGH/LOW
   ┌────────┐    │     ├──────────────┤
   │ Router │────┼────▶│ Dermatology  │  dermatology · LOW
   └────────┘    │     ├──────────────┤
        ▲        └────▶│ Neurology    │  neuro · HIGH
        │   route one case over A2A   └──────────────┘
   message/stream ─────────────┘
```

The point of going multi-container: **discovery is real**. The router learns the
graph at runtime from whoever answers. Stop a container and its edges disappear —
a case that needed it is refused, not misrouted.

> Runs with **no API key** on the default `mock` engine. Everything below is on a
> real wire, so every card and call is curl-able.

## Run the mesh

```bash
docker compose up --build
# open http://127.0.0.1:8000     (the router + the animated UI)
```

Type a presentation, hit **Discover & route**, and watch three acts:
**discover** (the router GETs each card, the graph builds) → **triage**
(sense → reason → act, a graph lookup) → **dispatch** (the matched container runs
its scoped tool loop and streams a disposition back over A2A).

## See the discovery for yourself

```bash
# each specialist serves its own Agent Card (its claims + skill + tool scope)
curl -s http://127.0.0.1:8101/.well-known/agent-card.json | python -m json.tool   # cardiology
curl -s http://127.0.0.1:8102/.well-known/agent-card.json | python -m json.tool   # pulmonology
curl -s http://127.0.0.1:8103/.well-known/agent-card.json | python -m json.tool   # dermatology
curl -s http://127.0.0.1:8104/.well-known/agent-card.json | python -m json.tool   # neurology

# the graph the ROUTER assembles purely from those cards:
curl -s http://127.0.0.1:8000/agents | python -m json.tool

# send a case straight to a specialist (the synchronous A2A form):
curl -s -X POST http://127.0.0.1:8101/a2a -H 'content-type: application/json' \
  -d '{"jsonrpc":"2.0","id":1,"method":"message/send",
       "params":{"message":{"role":"user","parts":[{"kind":"text",
                 "text":"chest tightness, left arm pain"}]}}}' | python -m json.tool
```

## Things to try in the UI

The animation is the same real `/run` either way — these just change what the
router does:

- **Transport: auto / stream / send.** `auto` reads each card's
  `capabilities.streaming` and picks the method itself. Dermatology ships with
  `streaming:false`, so routing a skin case auto-falls back to **`message/send`** —
  one request, one Task back, no intermediate frames. Force `stream`/`send` to
  compare. The **latency strip** (top right) shows first-byte vs total: streaming
  gives an early first frame; send's first byte == last byte.
- **Take a container down.** This demo is observe-only — the app never controls
  containers. Stop one in **Docker Desktop** (or `docker stop mesh-cardiology`),
  re-run a case, and the router's probe fails for real: that agent's edges vanish
  and a case that needed it is refused. Start it again and re-run to bring it back.
  See "kill a container" below.
- **Inject fault.** A tool throws inside the chosen container; the Task streams a
  **`failed`** status back and the router surfaces it instead of faking success.
- **Raw frames** (readout toggle). Flip from the narrated view to the literal
  `data:` SSE frames the browser received, in order — proof it's a real stream.

## The discovery demo: kill a container

The mesh is **observe-only** — nothing in the app stops containers. You do it, and
the UI reflects reality on the next run.

```bash
docker compose up -d                 # ALWAYS detached — a foreground `up` re-converges
                                     # the project and would undo your stops
docker stop mesh-dermatology         # or stop it in Docker Desktop
```
Re-run a skin case — the router probes dermatology, gets no answer, builds no
`dermatology · LOW` edge, and refuses. Bring it back:

```bash
docker start mesh-dermatology
```
Re-run; the edge returns. Nothing in the router changed — the graph is whatever
discovery finds alive at that moment. (Container names are pinned `mesh-<role>` so
`docker stop mesh-<role>` is easy.)

## What each container claims (discovered, not hardcoded)

| Container | Ward | Claims (system · acuity) | Tool scope |
|---|---|---|---|
| Cardiology | Cath Lab | cardiac · HIGH, cardiac · LOW | vitals, ecg, cath-lab, on-call, protocol |
| Pulmonology | ICU | respiratory · HIGH, respiratory · LOW | vitals, labs, imaging, beds, on-call, protocol |
| Dermatology | Clinic | dermatology · LOW | clinic, imaging, protocol |
| Neurology | Stroke Unit | neuro · HIGH | vitals, imaging, on-call, beds, protocol |

Deliberate gaps — **no container claims** `neuro · LOW`, `dermatology · HIGH`, or
anything in `gastro / ortho / general`. Cases mapping there are refused; you can
see the missing edge in the UI.

## A real engine (optional)

```bash
cp .env.example .env     # add ANTHROPIC_API_KEY or OPENAI_API_KEY
```
Pick that engine in the UI. The router's extraction and each specialist's loop use
it; the A2A protocol between them is unchanged.

## How the mesh maps to A2A

| In this demo | A2A concept |
|---|---|
| `GET /.well-known/agent-card.json` on each service | **Agent Card** discovery |
| the card's `x-triage-claims` | what the router turns into graph edges |
| `POST /a2a  message/stream` | **message/stream** (SSE task lifecycle) |
| `submitted → working → completed` | **Task** state machine |
| `artifact-update` → disposition | the **Artifact** returned to the router |
| service name as host (`cardiology:8000`) | one image, role via `AGENT_ROLE` |

---
*Schematic teaching software — canned clinical stubs, a labelled `mock` engine,
not medical software and not medical advice.*
