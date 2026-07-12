"""
embed.py — turns text into a vector with a small LOCAL model (no API key, no internet at runtime).

This is the ONE extra step vector data has that normal data does not:
    text  --(embedding model)-->  a fixed-length list of floats

We use BAAI/bge-small-en-v1.5 via fastembed (ONNX). It produces 384-dim vectors.
The model is downloaded once at image-build time (see Dockerfile) and cached, so the
running container needs no internet.
"""
from fastembed import TextEmbedding

MODEL_NAME = "BAAI/bge-small-en-v1.5"
DIM = 384

_model = None


def get_model():
    global _model
    if _model is None:
        _model = TextEmbedding(MODEL_NAME)
    return _model


def embed_texts(texts):
    """Return a list of numpy float32 vectors, one per input string."""
    return list(get_model().embed(list(texts)))


def embed_one(text):
    return embed_texts([text])[0]
