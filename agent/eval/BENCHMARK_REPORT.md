# Local model benchmark — Track 1

Bench box: Windows host, llama.cpp `n_threads=2` + `OMP_NUM_THREADS=2` (2-vCPU emulation), ctx 2048, temperature 0, thinking disabled. Peak RSS is the per-model subprocess peak working set — a proxy for the 4 GB cgroup limit of the grading box. Quality = offline heuristic judge over the 30 locally-served dev tasks (factual, math, sentiment, summarization, ner, logic); code categories route to Fireworks in production and do not weigh on local selection.

| model | size(GB) | load(s) | peakRAM(GB) | fits4GB | decode tok/s | est tok≤30s | overall pass | math | ner | sentiment | verdict |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Qwen3.5-4B | 2.74 | 6.2 | 3.59 | False | 4.7 | 96 | 0.97 | 1.0 | 1.0 | 1.0 | DISCARD: peak RSS 3.59 GB > 3.4 GB |
| SmolLM3-3B | 1.92 | 4.6 | 3.63 | False | 7.0 | 111 | 0.97 | 1.0 | 1.0 | 1.0 | DISCARD: peak RSS 3.63 GB > 3.4 GB |
| Phi-4-mini-instruct | 2.49 | 6.9 | 3.83 | False | 4.6 | 100 | 0.97 | 1.0 | 1.0 | 1.0 | DISCARD: peak RSS 3.83 GB > 3.4 GB |
| Llama-3.2-3B-Instruct | 2.02 | 6.0 | 3.75 | False | 6.0 | 144 | 0.93 | 1.0 | 1.0 | 1.0 | DISCARD: peak RSS 3.75 GB > 3.4 GB |
| Gemma-3-4B-it-QAT | 2.49 | 5.4 | 4.45 | False | 5.8 | 135 | 0.9 | 0.6 | 1.0 | 1.0 | DISCARD: peak RSS 4.45 GB > 3.4 GB; overall 0.90 not above baseline 0.90; regresses math vs baseline |
| Qwen2.5-0.5B | 0.49 | 2.7 | 0.63 | True | 20.6 | 566 | 0.9 | 0.8 | 1.0 | 1.0 | BASELINE |
| Qwen3.5-2B-Instruct | — | — | — | — | — | — | — | — | — | — | SKIPPED: No such model on Hugging Face (Qwen3.5 family starts at 4B); benchmarked Qwen3.5-4B as the nearest Qwen candidate instead. |

**Winner: Qwen2.5-0.5B** — overall pass 0.90, 20.6 decode tok/s, 0.49 GB, peak RSS 0.63 GB, load 2.7s.
**WARNING: no candidate passed every hard gate.** This is the best RAM-fitting model; the 30 s limit forces shorter outputs — lower `local_max_tokens_cap` accordingly.
