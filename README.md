# r1-distill

# Distilling DeepSeek-R1 Reasoning into Qwen 2.5

End-to-end pipeline for transferring long chain-of-thought (CoT) reasoning from
DeepSeek-R1 into open Qwen 2.5 models (7B and 32B) for self-hostable inference.

> **Status:** Work in progress. Track A (Qwen-7B) is the primary deliverable; Track B (Qwen-32B) follows.

## Training Data

| Source | Raw | Verified | Verification Rate |
|--------|-----|----------|-------------------|
| s1-1K  | 1000 | 1000 | N/A (curated upstream) |
| PCMB-500 (MMLU-Pro, decontaminated vs GPQA) | 500 | 413 | 82.6% |
| **Total** | **1500** | **1413** | — |

PCMB traces were generated using DeepSeek-R1 (`deepseek-reasoner`) as teacher,
with strict ground-truth verification: a trace is kept only if R1's final
answer letter matches the MMLU-Pro ground truth. Best-of-2 sampling was used
(temperature 0.6, top_p 0.95).

Generation cost: $0.57. Generation time: 38 minutes.

## Headline Results

| Model | GPQA Diamond | MATH-500 | AIME 2024 |
|-------|--------------|----------|-----------|
| Qwen2.5-7B-Instruct (baseline) | — | — | — |
| Qwen2.5-7B-R1-distill (ours) | — | — | — |
| Qwen2.5-32B-Instruct (baseline) | 41.9 | — | — |
| Qwen2.5-32B-R1-distill (ours) | **target: 63.5 (+21.6)** | — | — |

## Reproduce

```bash
make setup            # install deps
make generate-traces  # ~$50-100 in DeepSeek API costs
make curate           # build the 1.5K training set
make train-7b         # ~$80 on 4xA100 for ~3 hours on Runpod
make eval-7b          # GPQA + MATH-500 + AIME
make serve-7b         # vLLM server on port 8000
```

## License

Code: Apache 2.0. Models: subject to upstream Qwen and DeepSeek licenses.