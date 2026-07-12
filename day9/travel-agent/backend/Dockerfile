FROM python:3.12-slim

WORKDIR /app/backend

# Install deps first (better layer caching).
COPY backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the app + the static UI it serves.
COPY backend/ /app/backend/
COPY frontend/ /app/frontend/

EXPOSE 8000
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
