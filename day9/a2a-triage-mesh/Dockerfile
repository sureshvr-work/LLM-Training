FROM python:3.12-slim

WORKDIR /app

# deps first (better layer caching)
COPY app/requirements.txt ./app/requirements.txt
RUN pip install --no-cache-dir -r app/requirements.txt

# the one image that becomes any agent in the mesh
COPY app ./app
COPY frontend ./frontend

WORKDIR /app/app
EXPOSE 8000

# AGENT_ROLE (env) decides what this container is: router | cardiology |
# pulmonology | dermatology | neurology. Default engine is "mock" → no key.
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000"]
