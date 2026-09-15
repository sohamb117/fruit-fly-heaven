FROM python:3.12.11-slim-bookworm
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MUJOCO_GL=disable \
    MPLBACKEND=Agg \
    MPLCONFIGDIR=/tmp/cbbench-matplotlib \
    UV_LINK_MODE=copy
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates libgl1 libegl1 libglib2.0-0 libgomp1 fonts-dejavu-core \
    && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir uv==0.12.11
WORKDIR /bench
COPY pyproject.toml uv.lock ./
COPY connectome_body ./connectome_body
COPY configs ./configs
COPY scripts ./scripts
RUN uv sync --frozen --no-dev && uv run --frozen --no-dev cbbench bootstrap
ENV PATH=/bench/.venv/bin:$PATH
ENTRYPOINT ["cbbench"]
CMD ["data-status"]
