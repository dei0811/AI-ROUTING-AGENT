# Bundled local model weights (GGUF)

Weights are **not committed** (see `.gitignore`); they are downloaded here before
`docker build` and baked into the image. `config.json` (`local_model_path`) points
at the shipped file; `LOCAL_MODEL_PATH` overrides it. **Keep exactly one `.gguf`
here when building the image.**

## Shipped model: Phi-4-mini-instruct Q4_K_M (benchmark v2 winner, 2026-07-09)

Selected by `eval/bench_models.py` running every candidate inside a real
`linux/amd64` container capped like the grading box (`--memory=4g
--memory-swap=4g --cpus=2`) — full data in `eval/BENCHMARK_REPORT.md` /
`benchmark_results.json`:

- **Survived the full dev batch under 4 GB** (`State.OOMKilled=false`,
  cgroup peak 1.63 GB at ctx 1536 with q8_0 KV cache)
- 2.49 GB file, 4.0 s load, 5.1 decode tok/s in-container, ~93 tokens/30 s
- Dev-set pass (offline judge, 30 local tasks): **0.97 overall** —
  math 1.0, NER 1.0, sentiment 1.0, factual 1.0, logic 1.0, summarization 0.8
- MIT license

Shipped config: `local_ctx=1536`, `local_kv_type=q8_0`,
`local_max_tokens_cap=64` (60 % margin on the 30 s estimate — the grading
VM's shared vCPUs are slower than the bench container).

Runners-up that also passed every gate: Llama-3.2-3B (0.93, 6.6 tok/s) and
Qwen2.5-3B (0.93, math 0.8). SmolLM3-3B and Qwen3.5-4B matched Phi's 0.97
quality but could not fit 64 output tokens in 30 s at container speed.

History: bench v1 measured host RSS, which over-counts mmap'd weights — it
wrongly discarded every 3–4B model. v2's container runs also caught a real
production bug: `n_threads_batch` defaulting to `cpu_count()` stalls prefill
under a 2-vCPU quota (fixed in `local_model.py`).

Reproduce the shipped weights:

```bash
curl -L -o models/phi-4-mini.gguf \
  https://huggingface.co/unsloth/Phi-4-mini-instruct-GGUF/resolve/main/Phi-4-mini-instruct-Q4_K_M.gguf
```

Re-run the benchmark (re-downloads candidates into `models/bench/`, ~11 GB;
needs Docker with Linux containers): `python eval/bench_models.py`
(`AUTO_DELETE=0` stops before deleting).
