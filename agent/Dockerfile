# Judging VM runs linux/amd64 — build with --platform linux/amd64 (spec §1).
# Slim base keeps the compressed image far under the 10 GB cap and the
# cold start well under 60 s. No secrets baked in: FIREWORKS_API_KEY,
# FIREWORKS_BASE_URL and ALLOWED_MODELS are injected at runtime.
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Runtime files only; eval/ and tests/ are dev-time and stay out.
COPY main.py io_utils.py fireworks_client.py classify.py prompts.py \
     solve.py code_exec.py config.json ./

ENTRYPOINT ["python", "main.py"]
