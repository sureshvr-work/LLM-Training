"""
backend/main.py — Demo 3 controller (document -> structured JSON).

Routing:
  TXT/MD/CSV/JSON -> decoded to text          -> "text" mode
  PDF (digital)   -> text layer via PyMuPDF   -> "text" mode
  PDF (scanned)   -> NO text layer -> pages RASTERIZED to PNG -> "image" mode (vision)
  Image           -> passed through           -> "image" mode (vision)

System + user prompts and few-shot examples come from the UI.
Examples are forwarded to the LLM layer as real example turns.
"""
import base64
try:
    import pymupdf as fitz
except ImportError:
    import fitz
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from llm import extract

app = FastAPI()

MAX_PDF_PAGES = 5
RASTER_ZOOM = 2.0


class Example(BaseModel):
    doc: str
    out: str


class ExtractIn(BaseModel):
    interface: str = "raw"
    provider: str
    kind: str
    filename: str | None = None
    data_b64: str
    media_type: str | None = None
    system: str = ""
    instruction: str = ""
    examples: list[Example] = []


def pdf_text_and_count(raw: bytes):
    doc = fitz.open(stream=raw, filetype="pdf")
    text = "\n".join(p.get_text() for p in doc).strip()
    n = doc.page_count
    doc.close()
    return text, n


def pdf_to_pngs(raw: bytes, max_pages=MAX_PDF_PAGES, zoom=RASTER_ZOOM):
    doc = fitz.open(stream=raw, filetype="pdf")
    total = doc.page_count
    images = []
    for i, page in enumerate(doc):
        if i >= max_pages:
            break
        pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom))
        images.append({"b64": base64.b64encode(pix.tobytes("png")).decode(), "media": "image/png"})
    doc.close()
    return images, total


@app.get("/api/health")
async def health():
    return {"ok": True}


@app.post("/api/extract")
async def do_extract(body: ExtractIn):
    sys_p, usr_p = body.system, body.instruction
    ex = [{"doc": e.doc, "out": e.out} for e in body.examples]
    try:
        if body.kind == "pdf":
            raw = base64.b64decode(body.data_b64)
            text, pages = pdf_text_and_count(raw)
            if text:
                result = await extract.run(body.interface, body.provider, "text", sys_p, usr_p, text=text, examples=ex)
                result["source"] = {"kind": "pdf", "mode": "text", "fell_back": False, "pages": pages, "chars": len(text),
                                    "note": f"PDF had a text layer — extracted {len(text)} chars from {pages} page(s) with PyMuPDF and sent as text."}
                return result
            images, total = pdf_to_pngs(raw)
            if not images:
                raise HTTPException(422, "Could not read this PDF at all.")
            result = await extract.run(body.interface, body.provider, "image", sys_p, usr_p, images=images, examples=ex)
            result["source"] = {"kind": "pdf", "mode": "image", "fell_back": True, "pages": total, "rendered": len(images),
                                "note": f"No text layer found (scanned PDF). Rasterized {len(images)} of {total} page(s) to images and read them with vision OCR."}
            return result

        if body.kind == "txt":
            text = base64.b64decode(body.data_b64).decode("utf-8", "replace")
            result = await extract.run(body.interface, body.provider, "text", sys_p, usr_p, text=text, examples=ex)
            result["source"] = {"kind": "txt", "mode": "text", "fell_back": False, "chars": len(text),
                                "note": "Text file decoded and sent as text."}
            return result

        if body.kind == "img":
            images = [{"b64": body.data_b64, "media": body.media_type or "image/png"}]
            result = await extract.run(body.interface, body.provider, "image", sys_p, usr_p, images=images, examples=ex)
            result["source"] = {"kind": "img", "mode": "image", "fell_back": False,
                                "note": "Image sent natively to the model's vision."}
            return result

        raise HTTPException(400, f"unknown kind: {body.kind}")

    except HTTPException:
        raise
    except Exception as e:
        import httpx
        if isinstance(e, httpx.HTTPStatusError):
            raise HTTPException(e.response.status_code, e.response.text)
        raise HTTPException(502, f"{type(e).__name__}: {e}")
