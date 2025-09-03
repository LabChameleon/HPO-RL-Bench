# HPO-RL-Bench

[![OpenReview](https://img.shields.io/badge/OpenReview-Paper-blue)](https://openreview.net/forum?id=MlB61zPAeR)
[![HuggingFace](https://img.shields.io/badge/🤗%20HuggingFace-Dataset-yellow)](https://huggingface.co/datasets/gresashala/HPO-RL-Bench-data)

A tabular benchmark for hyperparameter optimization in reinforcement learning across Atari, MuJoCo, and classic control tasks.

## 📦 Where is the data?

All benchmark data is hosted on **Hugging Face Datasets**:

👉 **[https://huggingface.co/datasets/gresashala/HPO-RL-Bench-data](https://huggingface.co/datasets/gresashala/HPO-RL-Bench-data)**

The repo contains the `data_hpo_rl_bench/` folder used by this codebase.

---

## ⬇️ How to get the data

### Option A — Clone with Git LFS (recommended for full local copy)

```bash
# Install Git LFS once (if not already)
git lfs install

# Clone the dataset repo
git clone https://huggingface.co/datasets/gresashala/HPO-RL-Bench-data

# (Optional) Pull/checkout LFS objects explicitly if you ever see pointer files
cd HPO-RL-Bench-data
git lfs pull
git lfs checkout
cd ..
````

Now place/keep `HPO-RL-Bench-data/` **next to** this code repository and pass `data_path` to the handler—see below.

**Directory layout (excerpt):**

```
HPO-RL-Bench/                    # this code repo
HPO-RL-Bench-data/               # cloned from HF
└─ data_hpo_rl_bench/
   ├─ PPO/
   │  ├─ Pong-v0_0/              # seed shards by range
   │  ├─ Pong-v0_1/
   │  └─ Pong-v0_2/
   ├─ DQN/
   │  └─ ...
   └─ SAC/
      ├─ Hopper-v2_1/
      │  ├─ shard-0000/          # optional inner sharding to keep <10k files/dir
      │  └─ shard-0001/
      └─ ...
```

* `ENVIRONMENT_0` → seeds **0–2**
* `ENVIRONMENT_1` → seeds **3–5**
* `ENVIRONMENT_2` → seeds **6–9**
* Some heavy dirs (e.g., SAC Hopper/Humanoid) also contain `shard-0000/`, `shard-0001/`, … subfolders.

> If you just clone normally and see small text files starting with `version https://git-lfs.github.com/spec/v1`, run:
>
> ```bash
> git lfs pull
> git lfs checkout
> ```

### Option B — Programmatic access to individual files

If you only need a specific file:

```python
from huggingface_hub import hf_hub_download

p = hf_hub_download(
    repo_id="gresashala/HPO-RL-Bench-data",
    repo_type="dataset",
    filename="data_hpo_rl_bench/PPO/Pong-v0_0/Pong-v0_PPO_random_lr_-2_gamma_1.0_clip_0.2_seed4_eval.json",
)
print(p)  # local path to the downloaded file
```

---

## 🛠️ Install requirements

```bash
conda create -n hpo_rl_bench python=3.9 -y
conda activate hpo_rl_bench
conda install swig -y
pip install -r requirements.txt
```

> **Windows note (for pyrfr):** Microsoft Visual C++ 14.0+ is required.
> Install via [Microsoft C++ Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/).

---

## 🚀 Load and query the benchmark

If `HPO-RL-Bench-data/` sits **next to** this code repo, point `data_path` to the folder that **contains** `data_hpo_rl_bench/` (e.g., `"../HPO-RL-Bench-data"`).

```python
from benchmark_handler import BenchmarkHandler

# Example: static PPO on Atari Enduro, seed 0
benchmark = BenchmarkHandler(
    data_path="../HPO-RL-Bench-data",  # parent of `data_hpo_rl_bench/` (adjust if needed)
    environment="Enduro-v0",
    seed=0,
    search_space="PPO",
    set="static",
)

# Query a static configuration
configuration_to_query = {"lr": -6, "gamma": 0.8, "clip": 0.2}
queried = benchmark.get_metrics(configuration_to_query, budget=50)

# Query a dynamic configuration (multi-lr/gamma)
benchmark.set = "dynamic"
configuration_to_query = {"lr": [-3, -4], "gamma": [0.98, 0.99], "clip": [0.2, 0.2]}
queried_dyn = benchmark.get_metrics(configuration_to_query, budget=50)
```

---

## 📘 Further usage

See `benchmark-usages-examples.ipynb` for more examples, including extended sets and Bayesian optimization loops.

---

## 🖼️ Reproducing plots from the paper

* Figure 4: `plot_static_ppo.py`
* Figure 5a: `plot_dynamic.py`
* Figure 5b: `plot_extended.py`
* Figure 6a: `cd_diagram.py`
* Figure 6b: in `cd_diagram.py`, set `ALGORITHM="A2C"` (line \~384) and run
* Figures 2 & 3: `benchmark_EDA.py`

---

## 📝 Citation

If you use HPO-RL-Bench in your work, please cite:

```bibtex
@inproceedings{shala2024hporlbench,
  title     = {{HPO-RL-Bench}: A Zero-Cost Benchmark for Hyperparameter Optimization in Reinforcement Learning},
  author    = {Gresa Shala and Sebastian Pineda Arango and Andr{\'e} Biedenkapp and Frank Hutter and Josif Grabocka},
  booktitle = {Proceedings of the AutoML Conference 2024 (ABCD Track)},
  year      = {2024},
  url       = {https://openreview.net/forum?id=MlB61zPAeR}
}
```

---

