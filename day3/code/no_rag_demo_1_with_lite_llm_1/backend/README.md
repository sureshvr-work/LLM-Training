# LiteLLM Full-Context Demo (Demo 1 · OpenAI)

Chain: **frontend (nginx) → backend (FastAPI) → LiteLLM → OpenAI**
All same-origin (nginx proxies `/api/*` to the backend), so no CORS.

## Run
1. Put your key in `.env`:  `OPENAI_API_KEY=sk-...`
2. `docker compose up --build`
3. Open **http://localhost:8080**

## What it teaches
- Upload a doc/image → tiktoken token count shown instantly.
- Pick OpenAI (Anthropic/Google are stubbed "expand next").
- Edit the live request: system prompt, user message, temperature, max_tokens, top_p.
- Per question, choose **Ask from document** (full source, expensive) or
  **Ask from conversation** (history only, cheap — disabled on turn 1).
- Conversation history grows each turn; the MTU bar reacts live.
- Document is sent only when you ask *from document*; history stores text only.

## Expand to more models
Add an entry to `litellm/config.yaml` and enable the card in `frontend/index.html`
(`SOON` array). No backend changes — `complete()` already handles any model.

## Notes
- Each virtual key is capped at `max_budget: 1.0` in `backend/services.py`.
- `gpt-tokenizer` + `pdf.js` load from jsDelivr (needs internet); counts fall back
  to an approximation if offline.
- PDFs are sent as extracted text (reliable on OpenAI); images as data-URL image_url.
