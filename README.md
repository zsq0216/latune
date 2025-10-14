# LaTune: Lightweight & Adaptive Configuration Tuning for LLM Inference on Edge Devices

> A practical toolkit to **find fast, resource-aware runtime configs** for on-device LLM engines via **parameter selection**, **knowledge transfer**, and **two‑stage optimization**.

---

## TL;DR

* **Goal:** Maximize **throughput (TPS)** under a device’s **current resource budget** (VRAM/RAM/compute) for local LLM inference.
* **Key ideas:**

  1. **Focus** on the few parameters that really matter (parameter selection).
  2. **Warm‑start** from **good configs** found in **similar past tasks** (knowledge transfer).
  3. **Learn Pareto configs under stable resource conditions**, then **pick the best feasible one at runtime** (two‑stage optimization).
* **When to use:** You deploy LLMs on edge devices (PCs, embedded, laptops) and want **robust, high‑TPS** under **fluctuating resources**.

---

## Features

* 🔍 **Parameter selection**: importance analysis to shrink the search space.
* ♻️ **Knowledge transfer**: reuse top configs from similar (model × device × system_load) tasks.
* 🧠 **MOBO optimizer**: multi‑objective Bayesian optimization with EHVI to learn a **Pareto frontier** of (TPS, resource use).
* ⚡ **Runtime selector**: picks the **highest‑TPS feasible** config under the **current resource budget**.
* 📦 **Engine‑agnostic design** with reference adapter for `llama.cpp`.

---



## Requirements

* **Python** ≥ 3.10
* **OS**: Linux / macOS (Metal) / Windows (tested subset)
* **Optional accelerators**: CUDA‑capable NVIDIA GPU or Apple Silicon
* **Engines**: `llama.cpp` (reference adapter). Others can be added via a simple interface.
* **Build tools**: `cmake`, `gcc/clang` (if you build `llama.cpp` locally)

---

## Installation

```bash
# 1) Prepare the codebase
download and unzip the repository archive, or clone it from your online source.

# 2) (Optional) Create a clean Python environment
conda create -n latune_env python=3.10 -y
conda activate latune_env

# 3) Install Python dependencies
pip install -r requirements.txt

# 4) Build or install your inference engine (e.g., llama.cpp)
Place the inference engine directory (e.g., `llama.cpp/`) at the same level as the `latune` folder, and build it into an executable.

# 5) Download models
Store model files inside a folder named `models`, which should be located at the same level as `latune`.
For the following examples, you can download the model from:
https://huggingface.co/unsloth/Qwen3-4B-GGUF/blob/main/Qwen3-4B-Q4_0.gguf
```

Your directory structure should look like this:
```bash
project_root/
├── latune/
├── models/
│   └── qwen3-4B-q4.gguf
└── llama.cpp/
    └── build/
```
---


## Quick Start

Example: tune a 4B quantized model with `llama.cpp` on your hardware

### Step 1: Parameter Selection

```bash
python 00_params_ranking.py --hardware {your-hardware} --model qwen3-4b --quant q4
```

**Inputs:**

* `knobs_raw.json` in `knobs_files/` containing all tunable parameters (the existing file provides an example for `llama.cpp`).

**Outputs:**

* `.csv` files in `shap_outputs/{your-hardware}` recording historical observations.
* `.png` files in `shap_outputs/{your-hardware}` showing parameter importance rankings.
* `.json` files in `bounds/{your-hardware}` defining reference bounds.
* `.json` files in `knobs_files/{your-hardware}` listing ranked parameters.

---

### Step 2: Knowledge Transfer

```bash
python 01_meta_extraction.py --hardware {your-hardware} --model qwen3-4b --quant q4
```

**Outputs:**

* `meta_features/records.jsonl` containing meta‑features and matched top historical meta‑features.

---

### Step 3: Optimal Set Construction via MOBO

```bash
python 02_latune_workflow.py --hardware {your-hardware} --model qwen3-4b --quant q4
```

**Outputs:**

* `.pth` files in `surrogate_models/{your-hardware}` containing model checkpoints.
* `.json` files in `pareto_fronts/{your-hardware}` defining Pareto‑optimal configurations.
* `.json` files in `hv_progress/{your-hardware}` recording hypervolume progress.

Note: When there are no historical tuning results, knowledge transfer is ineffective and the system regresses to the cold-start state.
---

### Step 4: Select and Apply the Best Runtime Configuration

To emulate different system load levels, you can define sample configurations:

```python
resource_configs = [
    {"command": "python system_load_simulator.py", "resource": "low"},
    {"command": "python system_load_simulator.py --cpu 4 --memory 24576 --gpu-calc 1 --gpu-mem 4", "resource": "mid"},
    {"command": "python system_load_simulator.py --cpu 6 --memory 49152 --gpu-calc 2 --gpu-mem 6", "resource": "high"},
]
```

Then run the runtime adaptation:

```bash
python 04_adaptation_runtime.py --hardware {your-hardware} --model qwen3-4b --quant q4 --resource low
```

**Outputs:**

* `.json` files in `results/{your-hardware}` containing runtime results under varying resource budgets.

---

## Baseline Comparison

To evaluate LaTune against baseline methods:

```bash
python 03_baseline_workflow.py --hardware {your-hardware} --method {method} --model qwen3-4b --quant q4
```

Where `{method}` can be one of the following:

* `Default`
* `GA`
* `SCOOT`
* `CBO` (corresponding to *ReSTune* in the paper)

Then, to apply the runtime adaptation procedure for baselines:

```bash
python 04_adaptation_runtime.py --hardware {your-hardware} --model qwen3-4b --quant q4 --resource low
```

Finally, compare the results of LaTune with those of the baselines to assess performance differences.



## Acknowledgements

* `llama.cpp` and the broader on‑device LLM community.
* All baseline implementations referenced in the paper.
