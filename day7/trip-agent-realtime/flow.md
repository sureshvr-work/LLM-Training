# Trip Agent · Flow

## 1. High-level architecture

```mermaid
%%{init: {'theme': 'dark'}}%%
flowchart TD
    subgraph Browser["Browser"]
        FE["Frontend UI<br/>index.html"]
    end

    subgraph Backend["FastAPI Backend"]
        APP["app.py<br/>POST /run, SSE stream"]
        LOOP["loop.py<br/>sense, reason, act loop"]
        ENGINE["engine.py<br/>OpenAI / Anthropic adapter"]
        REG["registry.py<br/>tool validation and dispatch"]
        NOTIFY["notify.py<br/>email on final plan"]
        CFG["config.py<br/>keys and tunables"]
    end

    subgraph External["External Services"]
        LLMAPI["LLM API<br/>OpenAI / Anthropic"]
        TOOLS["Live tools<br/>geocode, weather, places, flights, websearch"]
        EMAIL["Resend email API"]
    end

    FE -->|"goal, engine"| APP
    APP -->|"SSE events"| FE
    APP --> LOOP
    LOOP <--> ENGINE
    ENGINE <--> LLMAPI
    LOOP <--> REG
    REG <--> TOOLS
    APP --> NOTIFY
    NOTIFY --> EMAIL
    CFG -.->|"keys, limits"| ENGINE
    CFG -.->|"keys, limits"| TOOLS
    CFG -.->|"keys, limits"| NOTIFY
```

Boxes group by where they run: **Browser** (just the UI), **Backend** (the loop is
the engine room, talking to the LLM adapter and the tool registry every turn),
**External** (the actual paid APIs). Dotted lines show `config.py` feeding
credentials/limits to anything that needs them.

## 2. Request flow (sequence)

```mermaid
%%{init: {'theme': 'dark'}}%%
sequenceDiagram
    participant FE as Frontend
    participant App as app.py
    participant AL as loop.run
    participant LLM as engine.py
    participant Reg as registry.py
    participant Notify as notify.py

    FE->>App: POST /run with goal and engine
    App->>AL: agent.run(goal, provider)
    AL->>FE: SSE start

    loop each turn, up to MAX_TURNS
        AL->>LLM: reason(history, specs)
        LLM-->>AL: decision, tool call or final
        AL->>FE: SSE reason event

        alt tool call present
            AL->>Reg: call(name, args)
            Reg-->>AL: result or error dict
            AL->>FE: SSE tool_result event
        else no tool call
            AL->>FE: SSE final event
        end
    end

    App->>Notify: notify_plan_ready(goal, plan_text, email)
    Notify-->>App: send status
```

## 3. Inside `loop.run()` — per-turn logic

```mermaid
%%{init: {'theme': 'dark'}}%%
flowchart TD
    A["Build history: system prompt and user goal"] --> B["yield start event"]
    B --> C{"turn within MAX_TURNS?"}
    C -- no --> Z["yield final: turn-limit message"]
    C -- yes --> D["REASON: provider.reason(history, specs)"]
    D -- exception --> E["yield error event, return"]
    D -- ok --> F["append assistant turn to history"]
    F --> G["yield reason event"]
    G --> H{"tool call present?"}
    H -- no --> I["yield final event with plan text, return"]
    H -- yes --> J["ACT: registry.call(name, args)"]
    J -- exception or HttpError --> K["result is error dict, ok = false"]
    J -- success --> L["ok = true, no error key"]
    K --> M["append tool result to history, capped at 4000 chars"]
    L --> M
    M --> N["yield tool_result event"]
    N --> C
```
