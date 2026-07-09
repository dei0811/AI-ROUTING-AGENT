# Bundled local model weights (GGUF)

Weights are **not committed** (see `.gitignore`); they are downloaded here before
`docker build` and baked into the image. `config.json` (`local_model_path`) points
at the shipped file; `LOCAL_MODEL_PATH` overrides it, and with neither set the
first `models/*.gguf` wins. **Keep exactly one `.gguf` here when building the
image** so the pick is deterministic and the image stays small.

## Shipped model: Qwen2.5-0.5B-Instruct Q4_K_M (benchmark winner, 2026-07-08)

Chosen by `eval/bench_models.py` under grading-box emulation (2 threads, ctx 2048,
temperature 0) — full data in `eval/BENCHMARK_REPORT.md` / `benchmark_results.json`:

- 0.49 GB file, **0.63 GB peak RSS**, 2.7 s load, 20.6 decode tok/s (~566 tokens/30 s)
- Dev-set pass (heuristic judge, 30 local tasks): **0.90 overall** — math 0.8,
  NER 1.0, sentiment 1.0, factual 1.0, logic 0.8, summarization 0.8

**Every 3B–4B candidate failed the 4 GB RAM hard gate** (peak RSS 3.59–4.45 GB vs
the 3.4 GB limit incl. agent headroom), despite Qwen3.5-4B / SmolLM3-3B /
Phi-4-mini scoring 0.97: quality is not the local bottleneck on this box — RAM is.
Consequences and future options:

- The 0.5B answers fast (~20 tok/s at 2 threads), so the 30 s per-task limit is
  comfortable; escalation to Fireworks covers what it gets wrong.
- To retry a bigger model, attack RAM first: Q4_0 instead of Q4_K_M, `n_ctx=1024`,
  smaller compute buffers — then re-run `python eval/bench_models.py`.
  SmolLM3-3B (3.63 GB peak, 0.97 pass, 7 tok/s) is the closest candidate.

Reproduce the shipped weights:

```bash
curl -L -o models/qwen2.5-0.5b-instruct-q4_k_m.gguf \
  https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF/resolve/main/qwen2.5-0.5b-instruct-q4_k_m.gguf
```

Re-run the benchmark (re-downloads candidates into `models/bench/`, ~11 GB):
see `eval/bench_models.py` docstring; `AUTO_DELETE=0` stops before deleting.
