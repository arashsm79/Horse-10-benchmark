# Use a Python image with uv pre-installed
FROM ghcr.io/astral-sh/uv:python3.11-bookworm-slim

# Install git and other dependencies
RUN apt-get update && apt-get install -y \
    git \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libgtk2.0-dev \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Install the project into `/app`
ADD . /app
WORKDIR /app
RUN uv sync --locked

RUN mkdir -p /data

# Place executables in the environment at the front of the path
ENV PATH="/app/.venv/bin:$PATH"
ENV PWD="/app"

# Set entrypoint to run the benchmark
ENTRYPOINT ["uv", "run", "horse10benchmark"]

# Default args that can be overridden
CMD ["data_dir=/data"]