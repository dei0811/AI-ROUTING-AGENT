# Local model benchmark v2 — container-based (Track 1)

Each candidate ran inside a `linux/amd64` container capped at `--memory=4g --memory-swap=4g --cpus=2` — the submission shape — so the RAM verdict is the kernel's OOM pass/fail (`State.OOMKilled`), not a host-RSS guess (v1's mistake). Footprint: ctx 1536 + q8_0 KV cache by default, one OOM retry at ctx 1024 + q4_0. Quality = offline heuristic judge over the 30 locally-served dev tasks; code categories route to Fireworks in production. Speed here is still optimistic vs the shared grading vCPUs — the shipped token cap takes a 60% margin on the 30 s estimate.

| model | size(GB) | load(s) | OOM@4g | cgroup peak(GB) | ctx/KV | decode tok/s | est tok≤30s | overall | math | ner | sentiment | verdict |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| SmolLM3-3B | 1.92 | 12.5 | False | 1.62 | 1536/q8_0 | 6.8 | 49 | 0.97 | 1.0 | 1.0 | 1.0 | DISCARD: only ~49 tokens fit in 30s (< 64) |
| Qwen3.5-4B | 2.74 | 11.1 | False | 1.71 | 1536/q8_0 | 4.2 | 56 | 0.97 | 1.0 | 1.0 | 1.0 | DISCARD: only ~56 tokens fit in 30s (< 64) |
| Phi-4-mini-instruct | 2.49 | 4.0 | False | 1.63 | 1536/q8_0 | 5.1 | 93 | 0.97 | 1.0 | 1.0 | 1.0 | KEEP |
| Llama-3.2-3B-Instruct | 2.02 | 5.0 | False | 1.64 | 1536/q8_0 | 6.6 | 140 | 0.93 | 1.0 | 1.0 | 1.0 | KEEP |
| Qwen2.5-3B-Instruct | 2.1 | 3.0 | False | 1.54 | 1536/q8_0 | 6.6 | 135 | 0.93 | 0.8 | 1.0 | 1.0 | KEEP |
| Qwen2.5-1.5B-Instruct | 1.12 | 5.8 | False | 0.82 | 1536/q8_0 | 11.9 | 300 | 0.83 | 1.0 | 0.4 | 1.0 | DISCARD: regresses ner vs baseline |
| Qwen2.5-0.5B | 0.49 | 0.9 | False | 0.23 | 1536/q8_0 | 29.1 | 803 | 0.83 | 0.6 | 1.0 | 1.0 | BASELINE |
| Gemma-3-4B-it-QAT | — | — | — | — | — | — | — | — | — | — | — | SKIPPED: excluded this round (spec §2): worst v1 RAM overflow and math regressed to 0.60 |

**Winner: Phi-4-mini-instruct** — overall pass 0.97, 5.1 decode tok/s in-container, 2.49 GB, survived the full batch under 4 GB (cgroup peak 1.63 GB, ctx 1536/q8_0 KV).
