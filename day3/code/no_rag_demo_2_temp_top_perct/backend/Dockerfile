# Backend image. Build context is the REPO ROOT (see docker-compose.yml)
# so we can copy both the backend code and the shared llm/ package.
FROM python:3.12-slim
WORKDIR /app
COPY backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY backend/main.py .
COPY llm/ ./llm/
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
