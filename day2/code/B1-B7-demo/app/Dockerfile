# The company app image.
FROM python:3.12-slim
WORKDIR /app

# Install just what the app needs.
RUN pip install --no-cache-dir httpx

COPY app.py .
CMD ["python", "app.py"]