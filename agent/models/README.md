# Bundled local model weights (GGUF)

Weights are **not committed** (see `.gitignore`); they are downloaded here before
`docker build` and baked into the image. `local_model.py` picks up the first
`models/*.gguf` automatically — or set `LOCAL_MODEL_PATH` / `local_model_path`
in `config.json` to choose explicitly. **Keep exactly one `.gguf` here when
building the image** so the pick is deterministic and the image stays small.

## Recommended weights (spec §4; grading box = 4 GB RAM, 2 vCPU, CPU-only)

| Role | Model | File size | Source |
|------|-------|-----------|--------|
| **Workhorse (submission default)** | Gemma 3 4B-it QAT Q4_0 | ~2.5 GB | `google/gemma-3-4b-it-qat-q4_0-gguf` (HF, license-gated) or `unsloth/gemma-3-4b-it-qat-GGUF` |
| Fast tier / fallback | Gemma 3 1B-it QAT Q4_0 | ~0.7 GB | `google/gemma-3-1b-it-qat-q4_0-gguf` or `unsloth/gemma-3-1b-it-qat-GGUF` |
| Non-Gemma alternative (math/code) | Qwen2.5-3B-Instruct Q4_K_M | ~1.9 GB | `Qwen/Qwen2.5-3B-Instruct-GGUF` |
| Dev smoke test only | Qwen2.5-0.5B-Instruct Q4_K_M | ~0.4 GB | `Qwen/Qwen2.5-0.5B-Instruct-GGUF` |

The 0.5B smoke model loads fast and proves the pipeline, but measurably fails
math/logic (seen on the dev set) — do not ship it. Decide the shipped model by
running `eval/run_eval.py` per candidate and comparing per-category pass rates.

## Download examples

```bash
# Dev smoke test (ungated):
curl -L -o models/qwen2.5-0.5b-instruct-q4_k_m.gguf \
  https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF/resolve/main/qwen2.5-0.5b-instruct-q4_k_m.gguf

# Workhorse (accept the Gemma license on HF first, then use an HF token):
huggingface-cli download google/gemma-3-4b-it-qat-q4_0-gguf gemma-3-4b-it-q4_0.gguf \
  --local-dir models/
```
