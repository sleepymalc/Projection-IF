# Projection-Regularization for Data Attribution at Scale

Experimental framework to validate theoretical claims about regularized sketching for Influence Functions.

## Setup

It's **not** required to follow the exact same steps in this section. But this is a verified environment setup flow that may help users to avoid most of the issues during the installation.

```bash
conda create -n hyproj python=3.10
conda activate hyproj

conda install -c "nvidia/label/cuda-12.4.0" cuda-toolkit
pip3 install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124

pip install sjlt --no-build-isolation
pip install dattri

pip install -r requirements.txt
```

## Experiments

### 1. Spectrum Bounds Validation (`spectrum_bounds.py`)

**Purpose:** Validate that sketched scores approximate exact scores within (1±ε) bounds.

**What it measures:**
- `ratio = sketched_score / exact_score` for test gradients
- `ε_95 = 95th percentile of |ratio - 1|`

**Usage:**
```bash
# Basic run
python spectrum_bounds.py --dataset mnist --model mlp

# With more trials
python spectrum_bounds.py --dataset mnist --model mlp --num_trials 10

# Different projection type
python spectrum_bounds.py --proj_type sjlt

# Large model (disk-cached gradients)
python spectrum_bounds.py --dataset maestro --model musictransformer --offload disk
```

### 2. Hyperparameter Selection (`hyperparam_selection.py`)

**Purpose:** Compare theory-driven vs utility-driven hyperparameter selection.

**What it measures:**
- LDS vs $\lambda$ and $m$
- Optimal $λ^{\ast}$ that maximizes LDS
- Whether $m \geq d_\lambda^{\ast}$ is sufficient for good LDS

**Two approaches:**
1. **Theory-driven:** Choose $m$ based on $d_\lambda : m \approx d_\lambda / \epsilon^2$
2. **Utility-driven:** Sweep $\lambda$ to maximize LDS, then verify $m \geq d_\lambda^{\ast}$

**Usage:**
```bash
python hyperparam_selection.py --dataset mnist --model mlp
```

## Command-Line Arguments

| Argument       | Description                                  | Default      |
| -------------- | -------------------------------------------- | ------------ |
| `--dataset`    | Dataset: mnist, cifar2, maestro              | mnist        |
| `--model`      | Model: lr, mlp, resnet9, musictransformer    | mlp          |
| `--proj_type`  | Projection: normal, rademacher, sjlt         | normal       |
| `--batch_size` | GPU batch size (tune for memory/utilization) | 32           |
| `--offload`    | Gradient storage: none, cpu, disk            | cpu          |
| `--cache_dir`  | Directory for disk cache                     | ./grad_cache |
| `--output_dir` | Directory for results                        | ./results    |
| `--seed`       | Random seed                                  | 42           |
| `--device`     | cuda or cpu                                  | cuda         |

### Additional Arguments for `spectrum_bounds.py`

| Argument         | Description                              | Default |
| ---------------- | ---------------------------------------- | ------- |
| `--num_trials`   | Number of trials per configuration       | 5       |
| `--min_m`        | Minimum projection dimension             | 1       |
| `--min_d_lambda` | Skip λ values where d_λ < this threshold | 5.0     |

## Model/Dataset Configurations

| Model            | Dataset | Parameters | Recommended Settings |
| ---------------- | ------- | ---------- | -------------------- |
| lr               | mnist   | ~8K        | `--offload none`     |
| mlp              | mnist   | ~100K      | `--offload cpu`      |
| resnet9          | cifar2  | ~6M        | `--offload cpu`      |
| musictransformer | maestro | ~13M       | `--offload disk`     |
