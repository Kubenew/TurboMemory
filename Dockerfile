FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml requirements.txt ./
RUN pip install --no-cache-dir -e ".[all]"

COPY . .

ENV PYTHONUNBUFFERED=1
ENV TURBOMEMORY_HOST=0.0.0.0
ENV TURBOMEMORY_PORT=8000

EXPOSE 8000

CMD ["python", "-m", "turbomemory.server.main"]