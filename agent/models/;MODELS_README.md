# Bundled Local Model Weights (GGUF)

Weights are **not committed** (see `.gitignore`); they are downloaded here before
`docker build` and baked into the image. `config.json` (`local_model_path`) points
to the bundled file, while `LOCAL_MODEL_PATH` overrides it. **Keep exactly one `.gguf`
file in this directory when building the image.**

## Bundled Model: Llama-3.2-3B-Instruct Q4_K_M (2026-07-09)

Selected after two rounds of evaluation (see `eval/BENCHMARK_REPORT.md` and
`eval/verify_image.py`):

- **Benchmark** (resource-capped container, 4 GB RAM / 2 vCPUs): 0.93 overall on
  the development set, with math, NER, and sentiment all scoring 1.0, 6.6 decode
  tokens/s, ~140 tokens per 30 seconds, and a cgroup peak memory usage of 1.64 GB
  using `ctx=1536` and `q8_0` KV cache.
- **Image verification** (cold container, image-bundled weights — matching the
  actual grading environment): all hard validation rules pass. Startup takes
  30–35 seconds, the slowest task completes in 24.0 seconds, the batch of the
  eight practice tasks finishes in 140 seconds, with no OOM, no empty responses,
  and a gzip-compressed image size of **1.94 GiB**.

Base configuration: `local_ctx=1536`, `local_kv_type=q8_0`. Token limits have
been tuned per category across versions—see the published images section below
for the current values.

**Why not the benchmark winner Phi-4-mini (0.97)?** Its 2.5 GB weight file left
no filesystem cache headroom within the 4 GB memory cgroup. Since mmap'd weights
are reclaimable clean pages, the kernel evicted them under memory pressure
(without triggering OOM), causing every prefill to page them back from disk.
Cold prefills reached ~20 seconds and produced empty or truncated responses.
Verification failed three separate times because of this behavior.

**`use_mmap=False` is critical:** model weights are loaded into resident anonymous
memory, which cannot be reclaimed when running with `--memory-swap=4g` (swap
disabled). This performs one bounded page-in during startup instead of repeated
page faults during inference, resulting in either fast and stable execution or a
clean OOM—never silent thrashing. Do **not** switch it back to speed up loading,
and do **not** enable `mlock` (the memlock ulimit is controlled by the grading
environment).

Reproduce the bundled model:

```bash
curl -L -o models/llama-3.2-3b.gguf \
  https://huggingface.co/bartowski/Llama-3.2-3B-Instruct-GGUF/resolve/main/Llama-3.2-3B-Instruct-Q4_K_M.gguf
```

Re-run the model benchmark (requires Docker and re-downloads approximately 11 GB):

```bash
python eval/bench_models.py
```

(`AUTO_DELETE=0` previews the process without deleting intermediate images.)

Re-verify the shipping image:

```bash
docker buildx build --platform linux/amd64 \
  -t track1-agent:verify --load .

python eval/verify_image.py
```

## Published Images

### v3 — Current (2026-07-12): Format and Pipeline Fixes After `ACCURACY_GATE_FAILED`

- **Reference:** `luis20072002/track1-agent:v3`
- **Digest:** `sha256:b5cf786c63d711bb205f7f0952684f0c7d4eb57af87e959a5cfefbda563654d2`
- **Pull:** `docker pull luis20072002/track1-agent:v3`
- **Registry:** Docker Hub (public, anonymous pull verified)
- **Token configuration:** `local_max_tokens_cap=120` (factual), `sentiment=48`;
  NER uses the compact output format with a cap of 84 tokens.

**Background**

Version v2 returned `ACCURACY_GATE_FAILED`. Local validation against the public
FAQ evaluation set showed that the failures were caused by **formatting and
pipeline issues rather than the local model itself**. The model successfully
completed every task where the pipeline allowed it to produce an answer.
Fixes were introduced in commit `207f83c`.

**Changes**

1. **Sentiment**
   - The prompt now requests both the sentiment label and a one-sentence
     explanation acknowledging both sides when appropriate.
   - The output cleaner no longer collapses responses into a single word or
     treats balanced explanations as malformed.
   - This also removed an unnecessary escalation to Fireworks.
   - Token cap increased from **8 → 48**.

2. **NER**
   - Switched to the compact output format:
     `type -> [strings]`
     (~45 tokens instead of ~120 for object-per-entity formatting).
   - `_extract_json` now recovers partially generated arrays, preserving
     complete extracted entities even when generation is truncated.
   - Token cap remains unchanged (84).

3. **Classifier**
   - Weak math cues such as `difference`, `solve`, `sum`, `average`, etc.
     now require the prompt to contain at least one numeric digit.
   - Conceptual comparison questions are now routed to `factual` instead of
     receiving the math prompt ("output only the number").
   - Two of the three factual FAQ examples previously failed because of this
     misclassification.

4. **Factual**
   - The prompt now allows concise explanations.
   - `local_max_tokens_cap` increased from **84 → 120**, preventing responses
     involving multiple concepts from being truncated midway.

**Validation After the Fixes**

- Real pipeline, real model weights, mocked Fireworks backend:
  - **8/8** locally verifiable practice tasks passed
    (previously 3/6 plus two routing errors).
  - Development set: **40/40**
  - Unit tests: **38/38 passing**

**Timing Note**

The higher token caps add approximately five seconds of decoding in the worst
case. Re-running `eval/verify_image.py` inside the resource-capped container
confirms that the 30-second-per-task requirement is still satisfied.

All four local categories, `use_mmap=False`, and the memory budget fixes remain
unchanged. Math and logic continue to be routed to Fireworks (their correctness
can only be fully validated through real remote execution).

---

### v2 — Previous (2026-07-12): Math & Logic Routed to Fireworks

- **Reference:** `luis20072002/track1-agent:v2`
- **Digest:** `sha256:5f6e75f33449ab4ab67c6779bd72c664bcaecfcc9ebaeedf99151a5167b9c93c`
- **Pull:** `docker pull luis20072002/track1-agent:v2`
- **Registry:** Docker Hub (public, anonymous pull verified)

**Changes Compared to v1 (Routing Table Only — No Hardcoded Models)**

The local execution of v1 failed:

- Practice-02 (math): produced **120** instead of **144**
- Practice-07 (logic): hallucinated `"Fish"`

These correspond to reasoning tasks where a local 3B model performs poorly.

`config.json` now routes both `math` and `logic` to the Fireworks **mid**
tier (general reasoning model, not the coding model), matching the existing
routing used for `code_debug` and `code_gen`.

The code-generation execution paths (`code_exec_categories` and
`local_code_exec_categories`) are now explicitly set to empty lists. If those
keys are removed, the application's default configuration automatically
re-enables code generation for math tasks.

The classifier also received additional logic cues such as:

- `"each ... owns ... one"`
- `"which ... does each"`
- `"either ... or"`

Previously, Practice-07 was classified as `factual`, preventing the rerouting
from taking effect.

Results:

- Development set: **40/40**
- Verification (mock backend):
  - **ALL RULES PASS**
  - 4 local tasks / 4 Fireworks tasks
  - Slowest task: 14.4 seconds
  - Batch runtime: 67 seconds
  - Image size: ~1.94 GiB

Superseded by **v3**.

---

### v1 — Stable Fallback

- **Reference:** `luis20072002/track1-agent:v1`
- **Digest:** `sha256:9b06fca24f2ac891b5d8f95aab26533f524d9e2a9f65ba5151fb4459a70c3a92`
- **Pull:** `docker pull luis20072002/track1-agent:v1`
- **Registry:** Docker Hub (public)

**Notes**

The image was built and published using:

```text
--provenance=false --sbom=false
```

(single-platform `linux/amd64` manifest).

Verification passes under:

```text
--memory=4g --cpus=2
```

This was the infrastructure submission (with Fireworks mocked). **v3** is the
currently active submission.
