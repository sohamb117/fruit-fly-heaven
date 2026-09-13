FROM python:3.11-slim-bookworm
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PORT=8080
WORKDIR /app
COPY requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt \
    && useradd --system --uid 10001 --no-create-home training
COPY server/ ./server/
COPY public/ ./public/
COPY public-manifest.json config.json build-manifest.json ./
USER 10001
EXPOSE 8080
CMD ["python", "-u", "/app/server/server.py"]
