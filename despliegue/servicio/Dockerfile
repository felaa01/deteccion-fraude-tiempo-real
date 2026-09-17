# Imagen oficial de uv + Python: evita instalar uv con pip y ya trae la versión de Python fijada.
FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

WORKDIR /app
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy

# Las dependencias se instalan antes de copiar el código: si solo cambia el código,
# Docker reusa esta capa y no vuelve a resolver/descargar paquetes (torch pesa ~190 MB).
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-install-project --no-dev

COPY src ./src
RUN uv sync --frozen --no-dev

ENV PATH="/app/.venv/bin:${PATH}"

EXPOSE 8000
CMD ["uvicorn", "fraude.servicio.principal:app", "--host", "0.0.0.0", "--port", "8000"]
