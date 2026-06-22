# Unlimited-OCR-ROCm Docker Image
# Supports ROCm 6.0+ on AMD Instinct / Radeon GPUs.
#
# Build:  docker build --build-arg ROCM_VERSION=6.2 -t unlimited-ocr-rocm .
# Run:    docker compose run --rm unlimited-ocr \
#           unlimited-ocr --image-dir /workspace/inputs --output-dir /workspace/outputs
#
# Supported ROCm versions: 6.0, 6.1, 6.2, 6.3

ARG ROCM_VERSION=6.2
FROM rocm/pytorch:rocm${ROCM_VERSION}_ubuntu22.04_py3.10_pytorch_2.5.0

LABEL org.opencontainers.image.title="Unlimited-OCR-ROCm"
LABEL org.opencontainers.image.description="Run Baidu Unlimited-OCR on AMD ROCm GPUs"
LABEL org.opencontainers.image.url="https://github.com/AIwork4me/Unlimited-OCR-ROCm"

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git curl libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Install SGLang with ROCm support
RUN pip install --no-cache-dir "sglang[all]>=0.4.0"

# Install project
WORKDIR /workspace
COPY pyproject.toml .
COPY src/ src/
RUN pip install --no-cache-dir -e .

ENTRYPOINT ["unlimited-ocr"]
CMD ["--help"]
