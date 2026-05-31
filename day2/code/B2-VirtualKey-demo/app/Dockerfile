# The company app image.
FROM python:3.12-slim

WORKDIR /app

RUN pip install --no-cache-dir httpx

COPY app.py .

CMD ["python", "app.py"]