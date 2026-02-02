# visualpath Dockerfile
# Multi-stage build for minimal image size
#
# Build: docker build -f docker/Dockerfile -t visualpath:latest .
# Run:   docker run --rm visualpath:latest version

# =============================================================================
# Stage 1: Builder
# =============================================================================
FROM python:3.11-slim AS builder

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install uv for faster package installation
RUN pip install --no-cache-dir uv

WORKDIR /build

# Copy the monolith structure needed for build
# Assumes build context is the visualpath directory
COPY . /build/visualpath/

# Also need visualbase as a dependency
# In production, this would come from PyPI or a registry
# For now, we install visualpath with dependencies mocked
WORKDIR /build/visualpath

# Create wheels
RUN uv pip compile pyproject.toml -o requirements.txt --extra cli --extra zmq 2>/dev/null || \
    pip wheel --no-cache-dir -w /wheels \
    pyyaml>=6.0 \
    pydantic>=2.0 \
    pyzmq>=25.0.0 \
    numpy>=1.24.0

# Build visualpath wheel
RUN pip wheel --no-cache-dir -w /wheels --no-deps .

# =============================================================================
# Stage 2: Runtime
# =============================================================================
FROM python:3.11-slim AS runtime

# Create non-root user for security
RUN useradd --create-home --shell /bin/bash visualpath

# Install runtime dependencies only
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy wheels from builder
COPY --from=builder /wheels /wheels

# Install packages
RUN pip install --no-cache-dir --no-index --find-links=/wheels \
    pyyaml pydantic pyzmq numpy \
    && rm -rf /wheels

# Install visualpath (last, as it may have issues without visualbase)
# In production, visualbase would also be installed
COPY --from=builder /wheels/visualpath*.whl /tmp/
RUN pip install --no-cache-dir --no-deps /tmp/visualpath*.whl || true \
    && rm -f /tmp/*.whl

# Set up working directory
WORKDIR /app

# Switch to non-root user
USER visualpath

# Default config directory
ENV VISUALPATH_CONFIG_DIR=/config
VOLUME ["/config"]

# Expose any ports if needed (e.g., for distributed workers)
# EXPOSE 5555

# Default entrypoint
ENTRYPOINT ["visualpath"]
CMD ["--help"]

# Labels
LABEL org.opencontainers.image.title="visualpath"
LABEL org.opencontainers.image.description="Video analysis pipeline platform"
LABEL org.opencontainers.image.source="https://github.com/your-org/visualpath"
