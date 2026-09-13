FROM python:3.13-slim

ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
WORKDIR /app

# The coordinator executes no model assets: it verifies the pinned config and
# distributes explicit parameters to contributors running the matching build.
COPY scripts/training_coordinator.py scripts/training_coordinator.py
COPY web/training/config.json web/training/config.json
RUN mkdir -p /data && chown 65532:65532 /data
USER 65532:65532

EXPOSE 7850
VOLUME ["/data"]
HEALTHCHECK --interval=30s --timeout=5s --start-period=5s --retries=3 \
  CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:7850/api/training/status', timeout=3).close()"]

CMD ["python", "scripts/training_coordinator.py", "--host", "0.0.0.0", "--port", "7850", "--config", "web/training/config.json", "--database", "/data/coordinator.sqlite3"]
