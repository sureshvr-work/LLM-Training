"""
backend/main.py — Provider Capability Lab controller.

One endpoint, /api/run, dispatches a (capability, provider) pair to the LLM
layer and returns the exact request(s), curl, live output, and notes.
"""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from llm import lab

app = FastAPI()


class RunIn(BaseModel):
    capability: str                 # text | pdf | image | audio | files
    provider: str                   # openai | anthropic | google | langchain
    kind: str | None = None
    filename: str | None = None
    data_b64: str | None = None
    media_type: str | None = None
    text: str | None = None
    prompt: str | None = None
    lc_target: str = "openai"


@app.get("/api/health")
async def health():
    return {"ok": True}


@app.post("/api/run")
async def run(body: RunIn):
    try:
        return await lab.run(
            body.capability, body.provider,
            kind=body.kind, b64=body.data_b64, media=body.media_type,
            text=body.text, prompt=body.prompt, lc_target=body.lc_target,
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(502, f"{type(e).__name__}: {e}")
