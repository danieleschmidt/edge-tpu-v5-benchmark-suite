FROM python:3.10-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    gnupg \
    software-properties-common \
    && rm -rf /var/lib/apt/lists/*

# Add Google Edge TPU repository
RUN echo "deb https://packages.cloud.google.com/apt coral-edgetpu-stable main" \
    | tee /etc/apt/sources.list.d/coral-edgetpu.list

# Add Google package signing key
RUN curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | apt-key add -

# Update package list and install Edge TPU runtime
RUN apt-get update && apt-get install -y \
    edgetpu-runtime-v5 \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY pyproject.toml .
COPY README.md .

# Install Python dependencies
RUN pip install --no-cache-dir -e .

# Copy source code
COPY src/ src/
COPY tests/ tests/

# Create non-root user for security
RUN useradd --create-home --shell /bin/bash tpuuser
RUN chown -R tpuuser:tpuuser /app
USER tpuuser

# Set Python path
ENV PYTHONPATH=/app/src

# Default command
CMD ["edge-tpu-v5-benchmark", "--help"]