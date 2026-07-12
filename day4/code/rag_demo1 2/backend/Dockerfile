# Built from the REPO ROOT (docker-compose.yml: context: .) so the image can
# copy BOTH the backend code AND the shared llm/ package.
FROM python:3.12-slim
WORKDIR /app

COPY backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# main.py
COPY backend/ /app/
# shared package the backend imports
COPY llm/ /app/llm/

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
