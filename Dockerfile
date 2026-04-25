# --- Stage 1: Downloader ---
FROM python:3.14.4-slim-bookworm AS downloader
ARG QUARTO_VERSION="1.8.26"
RUN apt-get update && apt-get install -y curl && \
    curl -o /tmp/quarto.deb -L https://github.com/quarto-dev/quarto-cli/releases/download/v${QUARTO_VERSION}/quarto-${QUARTO_VERSION}-linux-amd64.deb

# --- Stage 2: Final Image ---
FROM python:3.14.4-slim-bookworm

# 1. Get uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# 2. Install Quarto + Dependencies
COPY --from=downloader /tmp/quarto.deb /tmp/quarto.deb
RUN apt-get update && \
    (dpkg -i /tmp/quarto.deb || apt-get install -f -y) && \
    rm /tmp/quarto.deb && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 3. Install Python packages from pyproject.toml
# This ensures everyone gets the exact versions you pinned
COPY pyproject.toml uv.lock ./
RUN uv pip install --system --no-cache -r pyproject.toml

# 4. Copy App Code
COPY . .

EXPOSE 8000

CMD ["shiny", "run", "app.py", "--host", "0.0.0.0", "--port", "8000"]