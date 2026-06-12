# Provider Capability Lab

The same document-ingestion task across OpenAI, Anthropic, Google, and LangChain —
run it live, and see the exact request (raw REST + curl) that ran.

## Run
1. Copy `.env.example` to `.env` and add your three keys.
2. `docker compose up --build`
3. Open http://localhost:8091

## What it shows (v1)

| Capability | OpenAI | Anthropic | Google | LangChain |
|------------|--------|-----------|--------|-----------|
| Text       | Responses `input_text` | Messages text | `parts[].text` | unified HumanMessage |
| PDF        | `input_file` (base64) | `document` block | `inlineData` | content block |
| Image      | `input_image` | `image` block | `inlineData` | content block |
| Audio      | `/audio/transcriptions` | not native — transcribe first | `inlineData` (native) | Whisper loader |
| Files API  | `/v1/files` -> file_id | `/v1/files` (beta) -> file_id | File API (SDK) | loader into memory |

## Teaching flow
1. Text on all four — show the same task, three wire formats, one LangChain shape.
2. PDF / Image — flip providers, watch the request panel change shape.
3. Audio — Gemini native vs OpenAI transcription vs Anthropic's "transcribe first" cell.
4. Files API — the two-step upload-then-reference pattern (OpenAI / Anthropic live).

## Notes
- Requests/curl are constructed faithfully; live execution needs valid keys.
- Verify newest endpoint params (Responses `input_file`, Files API betas) against current docs.
- Port 8091 (Demo 1/3 use other ports) so labs can run side by side.
