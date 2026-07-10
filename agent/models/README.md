# Bundled local model weights (GGUF)

Weights are **not committed** (see `.gitignore`); they are downloaded here before
`docker build` and baked into the image. `config.json` (`local_model_path`) points
at the shipped file; `LOCAL_MODEL_PATH` overrides it. **Keep exactly one `.gguf`
here when building the image.**

## Shipped model: Llama-3.2-3B-Instruct Q4_K_M (2026-07-09)

Selected after two rounds of measurement (see `eval/BENCHMARK_REPORT.md` and
`eval/verify_image.py`):

- **Bench** (capped container, 4g/2cpu): 0.93 overall on the dev set with
  math / NER / sentiment all 1.0, 6.6 decode tok/s, ~140 tokens/30 s,
  cgroup peak 1.64 GB at ctx 1536 / q8_0 KV.
- **Image verification** (cold container, image-baked weights — the honest
  grading-box shape): ALL hard rules pass — startup 30–35 s, worst task 24.0 s,
  batch 140 s for the 8 practice tasks, no OOM, no empty answers, image
  **1.94 GiB** gzip-compressed.

Shipped config: `local_ctx=1536`, `local_kv_type=q8_0`, `local_max_tokens_cap=84`
(summarization capped to 64 to hold per-task margin).

**Why not the bench winner Phi-4-mini (0.97)?** Its 2.5 GB weights left no cache
slack under the 4 GB cgroup: mmap'd weights are reclaimable clean pages, the
kernel evicted them under pressure (never OOM), and every prefill re-faulted
them from disk — cold prefills ~20 s, empty truncated answers. Verification
failed three times on this physics.

**`use_mmap=False` is load-bearing:** weights load as resident anonymous memory,
which cannot be evicted with `--memory-swap=4g` (no swap). One bounded fault-in
at startup replaces unbounded re-faulting during tasks — fast-and-stable or a
clean OOM, never silent thrash. Do not revert it to speed up load; do not use
`mlock` (the memlock ulimit belongs to the grading harness).

Reproduce the shipped weights:

```bash
curl -L -o models/llama-3.2-3b.gguf \
  https://huggingface.co/bartowski/Llama-3.2-3B-Instruct-GGUF/resolve/main/Llama-3.2-3B-Instruct-Q4_K_M.gguf
```

Re-run the model benchmark (needs Docker, re-downloads ~11 GB):
`python eval/bench_models.py` (`AUTO_DELETE=0` previews without deleting).
Re-verify the shipping image: `docker buildx build --platform linux/amd64
-t track1-agent:verify --load .` then `python eval/verify_image.py`.
