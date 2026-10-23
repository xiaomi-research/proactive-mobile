# [CVPR 2026] ProactiveMobile: A Comprehensive Benchmark for Boosting Proactive Intelligence on Mobile Devices

<div align="center">

[![Paper](https://img.shields.io/badge/Paper-arXiv:2602.21858-red)](https://arxiv.org/abs/2602.21858)
[![Dataset](https://img.shields.io/badge/🤗%20Dataset-ProactiveMobile-yellow)](https://huggingface.co/datasets/xiaomi-research/ProactiveMobile)
[![License](https://img.shields.io/badge/License-CC%20BY--NC--SA%204.0-green.svg)](https://creativecommons.org/licenses/by-nc-sa/4.0/)

</div>

---

## Overview

**ProactiveMobile** is an executable benchmark for **proactive intelligence in mobile agents** — moving beyond the reactive paradigm (passively executing user commands) toward agents that **anticipate user needs and act on their own**.

The task: infer latent user intent from four dimensions of on-device context, then generate an executable function sequence from a unified function pool.

- 📱 **14** real-world scenarios
- 📊 **3,658** instances with multi-answer (1–3 target actions) annotations
- 🔧 **61** executable APIs
- 🧭 Four context dimensions: **User Profile · Device Status · World Information · Behavioral Trajectories**

> **Note on the function pool.** For this release, semantically overlapping functions were consolidated, and the pool was pruned to the **61 APIs actually used** by the benchmark.

## Dataset

🤗 [**xiaomi-research/ProactiveMobile**](https://huggingface.co/datasets/xiaomi-research/ProactiveMobile)

```python
from datasets import load_dataset

dataset = load_dataset("xiaomi-research/ProactiveMobile")
```

## Model

The fine-tuned proactive agent (Qwen2.5-VL-7B-Instruct + Proactive) will be released on 🤗 Hugging Face. *(Link coming soon.)*

## Inference & Evaluation

`evaluation/inference_and_eval.py` runs local [vLLM](https://github.com/vllm-project/vllm)
inference over the benchmark and scores it. It is a single, dependency-light file
(standard library + `vllm` + `Pillow`).

### Install

```bash
pip install vllm pillow
```

### Data format

Each instance provides four context dimensions under `reference_information`
(`profile` / `phone` / `world` / `trace`) and one or more gold `recommendations`.
`trace` is either a list of screenshot steps (**GUI** data) or a natural-language
operation sequence (**TEXT** data). The model is asked to emit a three-part output:

```
<think>reasoning</think><rec>natural-language recommendation</rec><function>{"model_recommendation": [...]}</function>
```

Only the `<function>` part is scored.

### Step 1 — inference

```bash
python evaluation/inference_and_eval.py \
    --mode eval \
    --model_path <path-or-hf-id-of-the-model> \
    --eval_file   benchmark_zh.json \
    --function_pool_file function_pool.json \
    --output_format thinking recommendation \
    --tensor_parallel_size 4 --pipeline_parallel_size 2 \
    --output_path ./result
```

For GUI data, screenshot paths in `trace` must resolve on disk. If your dataset
stores relative paths (e.g. `Benchmark/....png`), map them to your local image
root with `--image_prefix Benchmark/ --image_root /your/image/root`.

> **GPU layout.** vLLM uses tensor parallelism; Qwen2.5-VL-7B has 28 attention
> heads, so `--tensor_parallel_size` must divide 28 (use 1/2/4/7/14). To use 8
> GPUs, combine tensor + pipeline parallel, e.g. `--tensor_parallel_size 4
> --pipeline_parallel_size 2`.

### Step 2 — compute metrics

```bash
python evaluation/inference_and_eval.py \
    --mode compute \
    --model_path <path-or-hf-id-of-the-model> \
    --eval_file   benchmark_zh.json \
    --function_pool_file function_pool.json \
    --output_path ./result
```

This prints and logs, broken down by **difficulty (L1/L2/L3) × modality (GUI/TEXT)**:

| Metric | Definition |
|--------|------------|
| **Type-Acc** | function-name sequence exact match (order + length) against the best-matching gold |
| **Precision / Recall / F1** | set-based overlap of function names vs. the best-matching gold |
| **FTR** (False Trigger Rate) | on *no-action* instances (gold is empty), the fraction where the model nonetheless predicted an action — lower is better |

### Sampling: greedy (temperature = 0) is recommended

The script **defaults to greedy decoding (`--temperature 0`)**, which we recommend
for evaluation: the output is deterministic and fully reproducible (identical
results across runs), and avoids the run-to-run variance that `temperature = 1.0`
introduces — especially in FTR, which is sensitive to borderline no-action cases.
The paper reports numbers at `temperature = 1.0`; pass `--temperature 1.0 --top_p 0.7`
to reproduce the exact paper setting.

### Reference results (temperature = 0, greedy)

Qwen2.5-VL-7B-Instruct + Proactive, on the full benchmark (3,658 instances):

| Split | Type-Acc | Precision | Recall | F1 | FTR↓ |
|-------|:-------:|:--------:|:------:|:----:|:----:|
| **Overall** | 49.29 | 54.23 | 54.19 | 53.83 | 8.86 |
| &nbsp;&nbsp;L1 | 72.41 | 75.20 | 75.33 | 75.16 | 6.98 |
| &nbsp;&nbsp;L2 | 57.93 | 63.52 | 63.44 | 63.10 | 7.12 |
| &nbsp;&nbsp;L3 | 38.99 | 43.90 | 43.85 | 43.45 | 11.23 |
| **TEXT** | 64.04 | 65.60 | 65.58 | 65.46 | 1.31 |
| **GUI** | 34.57 | 42.88 | 42.81 | 42.23 | 24.37 |

All scores are percentages. Numbers are deterministic under greedy decoding.

## Citation

```bibtex
@article{kong2026proactivemobile,
  title   = {ProactiveMobile: A Comprehensive Benchmark for Boosting Proactive Intelligence on Mobile Devices},
  author  = {Kong, Dezhi and Feng, Zhengzhao and Liang, Qiliang and others},
  journal = {arXiv preprint arXiv:2602.21858},
  year    = {2026}
}
```

## License

Released under [**CC BY-NC-SA 4.0**](https://creativecommons.org/licenses/by-nc-sa/4.0/) — non-commercial use, attribution required, share-alike.
