"""
backend/main.py — thin controller.

It owns the HTTP API. It does NO LLM logic itself — it delegates to the
llm package. The frontend (nginx) serves the page and proxies /api/* here.

Flow:
  browser  ->  nginx  ->  POST /api/chat {interface, provider, system, user, ...}
            ->  pick interface: raw REST (providers) or LangChain
            ->  <module>.run()  ->  the call for that interface + provider
            ->  returns {request sent, response received, answer}
            ->  back to the browser, so the student sees both directions.

The two interfaces expose the SAME run() signature and return the SAME
envelope, so routing between them is a one-line choice.
"""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import httpx

from llm import providers, langchain_iface

INTERFACES = {
    "raw": providers,            # raw REST — branches on everything
    "langchain": langchain_iface,  # LangChain — branches on one string
}

app = FastAPI()


class ChatIn(BaseModel):
    interface: str = "raw"              # "raw" | "langchain"
    provider: str                       # "openai" | "anthropic" | "google"
    system: str | None = None
    user: str
    temperature: float | None = None
    max_tokens: int | None = None
    top_p: float | None = None


@app.get("/api/health")
async def health():
    return {"ok": True}


@app.post("/api/chat")
async def chat(body: ChatIn):
    mod = INTERFACES.get(body.interface)
    if mod is None:
        raise HTTPException(400, f"unknown interface: {body.interface}")
    try:
        result = await mod.run(
            body.provider, body.system, body.user,
            body.temperature, body.max_tokens, body.top_p,
        )
        result.setdefault("interface", "Raw REST")   # raw module doesn't set it
        return result
    except httpx.HTTPStatusError as e:
        # Surface the real provider error so students see what failed.
        raise HTTPException(e.response.status_code, e.response.text)
    except (httpx.HTTPError, ValueError) as e:
        raise HTTPException(502, str(e))
    except Exception as e:
        # LangChain wraps provider errors in its own exception types.
        raise HTTPException(502, f"{type(e).__name__}: {e}")
