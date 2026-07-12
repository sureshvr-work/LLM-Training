# The guardrail gateway image - B7 edition
FROM python:3.12-slim

WORKDIR /app

# Install gateway dependencies including B7 response controls
RUN pip install --no-cache-dir \
    fastapi \
    uvicorn \
    httpx \
    pydantic \
    presidio-analyzer \
    presidio-anonymizer \
    spacy && \
    python -m spacy download en_core_web_lg

COPY gateway.py .

EXPOSE 8000

CMD ["uvicorn", "gateway:app", "--host", "0.0.0.0", "--port", "8000"]