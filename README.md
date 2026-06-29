# ProactiveMobile: A Comprehensive Benchmark for Boosting Proactive Intelligence on Mobile Devices

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
