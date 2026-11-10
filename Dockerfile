FROM python:3.12-slim
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements.txt requirements-serve.txt ./
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -r requirements-serve.txt -c requirements.txt
COPY configs/ configs/
COPY src/ src/
COPY models/champion/ models/champion/
COPY models/reference.json models/
RUN mkdir -p logs
