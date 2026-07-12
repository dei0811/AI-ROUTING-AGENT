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

## Published images

### v2 — current (2026-07-12): math & logic rerouted to Fireworks

- **Ref:** `luis20072002/track1-agent:v2`
- **Digest:** `sha256:5f6e75f33449ab4ab67c6779bd72c664bcaecfcc9ebaeedf99151a5167b9c93c`
- **Pull:** `docker pull luis20072002/track1-agent:v2`
- **Registro:** Docker Hub (público, pull anónimo verificado)
- **Cambio vs v1 (routing-table, sin hardcodear modelos):** el run local de v1
  falló practice-02 (math: 120 en vez de 144) y practice-07 (logic: inventó
  "Fish") — las categorías de razonamiento donde un modelo local 3B es débil.
  `config.json` ahora rutea `math` y `logic` a Fireworks en tier **mid**
  (general/razonamiento, no el modelo de código), igual que ya iban
  `code_debug`/`code_gen`; los paths emit-code (`code_exec_categories` y
  `local_code_exec_categories`) quedaron en listas vacías **explícitas** (si
  se borra la clave, los defaults del código reactivan emit-code para math).
  El clasificador ganó cues de logic ("each ... own(s) ... one",
  "which ... does each", "either ... or"): practice-07 caía en `factual` y el
  reroute no lo alcanzaba; dev set 40/40. Las cuatro categorías locales
  (factual, sentiment, summarization, ner), `use_mmap=False` y los budget
  fixes quedan intactos. Verify (mock): ALL RULES PASS, 4 local / 4 fireworks,
  peor task 14.4s, batch 67s, ~1.94 GiB.

### v1 — fallback estable

- **Ref:** `luis20072002/track1-agent:v1`
- **Digest:** `sha256:9b06fca24f2ac891b5d8f95aab26533f524d9e2a9f65ba5151fb4459a70c3a92`
- **Pull:** `docker pull luis20072002/track1-agent:v1`
- **Registro:** Docker Hub (público)
- **Nota:** imagen construida y publicada con `--provenance=false --sbom=false`
  (manifest linux/amd64 de plataforma única). Verify pasa bajo `--memory=4g
  --cpus=2`. Submission de infraestructura (Fireworks en mock); v2 la
  reemplaza como submission activa.
