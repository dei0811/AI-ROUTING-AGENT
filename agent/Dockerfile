# Judging VM runs linux/amd64 — build with --platform linux/amd64 (spec §1).
# Slim base + one GGUF keeps the compressed image far under the 10 GB cap
# and the cold start well under 60 s (the GGUF loads via mmap in seconds).
# No secrets baked in: FIREWORKS_API_KEY, FIREWORKS_BASE_URL and
# ALLOWED_MODELS are injected at runtime.
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

COPY requirements.txt .
# Prefer the prebuilt CPU wheel for llama-cpp-python; fall back to a
# source build (build tools purged in the same layer). GGML_NATIVE=OFF
# is required for the fallback: the grading CPU differs from the build
# machine, and -march=native binaries would die with illegal-instruction.
RUN pip install --no-cache-dir --only-binary=:all: -r requirements.txt \
      --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu \
 || ( apt-get update \
      && apt-get install -y --no-install-recommends build-essential cmake \
      && CMAKE_ARGS="-DGGML_NATIVE=OFF" \
         pip install --no-cache-dir -r requirements.txt \
      && apt-get purge -y build-essential cmake \
      && apt-get autoremove -y \
      && rm -rf /var/lib/apt/lists/* )

# Runtime files only; eval/ and tests/ are dev-time and stay out
# (see .dockerignore). models/ must hold exactly one .gguf when building.
COPY main.py io_utils.py fireworks_client.py local_model.py classify.py \
     prompts.py solve.py code_exec.py config.json ./
COPY models/ ./models/

ENTRYPOINT ["python", "main.py"]
