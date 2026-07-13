# Track 1 — Hybrid Token-Efficient AI Agent

A general-purpose AI agent for the **AMD Developer Hackathon: ACT II — Track 1**. It solves a
batch of natural-language tasks across **8 capability categories** while spending as few
**Fireworks AI** tokens as possible: most work is answered by a bundled **local model** (which
costs zero toward the score), and only the categories that genuinely need it are escalated to
Fireworks.

> **This is a batch agent, not a web app.** It reads `/input/tasks.json`, solves every task, and
> writes `/output/results.json`. There is no visual demo — run the Docker image (below).

## Docker image

```bash
docker pull luis20072002/track1-agent:v3
```

- Registry: **Docker Hub**, public, `linux/amd64`, ~1.94 GiB compressed.
- The judging harness injects `FIREWORKS_API_KEY`, `FIREWORKS_BASE_URL`, and `ALLOWED_MODELS` at
  runtime — the image reads them from the environment. No secrets or model IDs are baked in.

## How it works

```
/input/tasks.json
      │
      ▼
  classify category  ──►  ROUTE
      │                     │
      │        ┌────────────┴────────────┐
      ▼        ▼                         ▼
   LOCAL — Llama-3.2-3B (CPU)      FIREWORKS — allowed models
   0 tokens · resident weights     counted tokens · via base URL
      │                         │
      └───────────┬─────────────┘
                  ▼
   validate  ──►  /output/results.json
```

**Local (free):** factual knowledge · sentiment · summarization · named-entity recognition.
**Fireworks (counted):** mathematical reasoning · logical reasoning · code debugging · code generation.

The routing intelligence is the point: a small 3B model handles the short-answer categories
reliably for zero tokens, and remote calls are spent only where accuracy demands them.

## Key engineering

- **Runs in the grading box:** 4 GB RAM · 2 vCPU · CPU-only · `linux/amd64` · start < 60 s ·
  < 30 s per task · batch < 10 min · image ≤ 10 GB.
- **`use_mmap=False` (load-bearing):** the local model's weights load as resident memory that the
  4 GB cgroup cannot evict, replacing per-task disk re-faulting with one bounded fault-in at
  startup. See [`agent/models/README.md`](agent/models/README.md) for the full rationale and the
  model-selection benchmark.
- **Time-budget guard** keeps every task under 30 s and the batch under 10 min.

## Repository layout

```
agent/
  main.py            entrypoint (reads /input, writes /output)
  solve.py           per-task routing & solving
  classify.py        heuristic category detection
  local_model.py     llama-cpp-python wrapper (use_mmap=False)
  fireworks_client.py OpenAI-compatible client (+ mock mode)
  prompts.py         per-category prompts
  config.json        routes, caps, model tiers
  eval/              verification & benchmark harness
  models/            bundled GGUF (gitignored) + technical README
  Dockerfile
```

## Verify locally

```bash
cd agent
# download the weights (gitignored) per agent/models/README.md, then:
docker build -t track1-agent:verify .
docker run --rm --memory=4g --memory-swap=4g --cpus=2 \
  -e MOCK_FIREWORKS=1 -e FIREWORKS_API_KEY=dummy \
  -e FIREWORKS_BASE_URL=http://localhost -e ALLOWED_MODELS=dummy-model \
  -v "$PWD/practice:/input:ro" -v "$PWD/out:/output" track1-agent:verify
python eval/verify_image.py
```

## Stack

Python · llama.cpp (`llama-cpp-python`) · Fireworks AI (OpenAI-compatible API) · Docker.
