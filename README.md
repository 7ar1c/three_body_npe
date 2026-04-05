# Three-Body Neural Posterior Estimation (NPE)

Bayesian inference for the planar three-body problem using simulation-based inference (`sbi`) and Neural Posterior Estimation (NPE).

The project generates simulated orbital tracks, trains a posterior estimator for initial conditions and masses, and evaluates calibration/coverage diagnostics.

## What this repo does

- Simulates 2D three-body trajectories using `scipy.integrate.solve_ivp`.
- Builds observation tensors with shape `(32, 13)`:
	- 12 dynamical features (`x/y/vx/vy` for 3 bodies)
	- 1 time column
- Learns posterior $p(\theta \mid x)$ for a 15D parameter vector:
	- positions: `x1,y1,x2,y2,x3,y3`
	- velocities: `vx1,vy1,vx2,vy2,vx3,vy3`
	- masses: `m1,m2,m3`
- Evaluates uncertainty quality with:
	- held-out error metrics,
	- posterior predictive checks (PPC),
	- simulation-based calibration (SBC),
	- expected coverage (TARP),
	- local coverage (L-C2ST).

## Repository layout

```
.
├── main.py                     # small end-to-end demo: generate + train + evaluate
├── npe.py                      # NPE model/config/training and posterior utilities
├── validation.py               # PPC/SBC/TARP/L-C2ST diagnostics
├── train/
│   └── train.py                # training pipeline
├── test/
│   ├── evaluate.py             # CPU-focused pretrained model evaluation
│   ├── evaluate_cuda.py        # GPU-focused pretrained model evaluation
│   └── test_on_one.py          # inspect one random test example
├── generate/
│   ├── generate_data.py        # 3-body ODE + late-time track sampling
│   ├── generate_cluster.py     # regular dataset batch generation
│   ├── generate_chaotic_cluster.py     # chaotic dataset match generation
│   └── generate_close_encounter.py
├── cluster_scripts/            # SLURM scripts
└── final_results/              # final results
```

## Requirements

- Python 3.12
- macOS/Linux
- Optional GPU for large-scale training/evaluation (`cuda` or Apple `mps`)

Install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

## Data format

Each dataset `.npz` file must contain:

- `theta`: shape `(N, 15)`
- `x`: shape `(N, 32, 13)`

Validation is enforced in `npe.load_dataset()`.


## Large-scale workflow

### 1) Generate data

Regular batches:

```bash
python generate/generate_cluster.py
```

Chaotic close-encounter batches:

```bash
python generate/generate_chaotic_cluster.py
```

### 2) Train model on shards

```bash
python -m "train.py"
```

This saves the model like `trained_npe_model_XXXk.pt`.

### 3) Evaluate pretrained model


```bash
python -m "evaluate_cuda.py"
```

Outputs are written to `plots/`.

## Cluster usage (SLURM)

Scripts are in `cluster_scripts/`:

- `generate.sh`
- `train.sh`
- `evaluate.sh`
