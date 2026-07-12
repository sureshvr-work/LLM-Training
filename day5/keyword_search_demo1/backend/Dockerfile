# Built from the REPO ROOT (docker-compose.yml: context: .) so the image can
# copy BOTH the backend code (main.py + engine.py) AND the shared llm/ package.
FROM python:3.12-slim
WORKDIR /app

COPY backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# backend code: main.py (FastAPI wiring) + engine.py (pure keyword engine)
COPY backend/ /app/
# the only place that talks to a model provider (used for the answer step)
COPY llm/ /app/llm/

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
