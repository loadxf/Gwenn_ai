FROM python:3.11-slim

WORKDIR /app

# System deps for opencv-python-headless and chromadb
RUN apt-get update && \
    apt-get install -y --no-install-recommends libgl1 libglib2.0-0 && \
    rm -rf /var/lib/apt/lists/*

# Install uv and sync dependencies (source not yet available)
COPY pyproject.toml uv.lock ./
RUN pip install --no-cache-dir uv && uv sync --no-dev --frozen --no-install-project

# Copy application code and install the project
COPY gwenn/ ./gwenn/
COPY gwenn_skills/ ./gwenn_skills/
RUN uv sync --no-dev --frozen

# Persistent data volume
VOLUME /app/gwenn_data

# Run as non-root user
RUN useradd -m gwenn && chown -R gwenn:gwenn /app
USER gwenn

ENV PYTHONUNBUFFERED=1

# Health check via daemon socket
HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD python -c "import json,socket,sys;s=socket.socket(socket.AF_UNIX);s.connect('/app/gwenn_data/gwenn.sock');s.sendall((json.dumps({'type':'ping'})+'\n').encode());sys.exit(0)" || exit 1

ENTRYPOINT ["uv", "run", "gwenn-daemon"]
