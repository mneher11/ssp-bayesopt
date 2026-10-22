<div align="center">

# SSP-BO: Efficient Active Exploration with Compositional Neurosymbolic Representations

[![Python](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-passing-brightgreen.svg)]()

</div>

---

## About

**SSP-BO** is a sample-, memory-, and time-efficient Bayesian optimization (BO) library built on compositional neurosymbolic representations — Spatial Semantic Pointers (SSPs). It replaces the Gaussian process surrogate of standard BO with a **Bayesian linear regression (BLR)** model over fixed-dimensional vector-symbolic embeddings.

Standard BO requires *O(n³)* time and *O(n²)* memory as the number of observations *n* grows. SSP-BO reduces this to *O(d³)* and *O(d²)* in the fixed embedding dimension *d* — giving **constant** time and memory per sample-selection step, independent of how many samples have been collected. On standard benchmarks, this yields up to **60–200× faster** sample selection with no loss in accuracy.

<div align="center">
<img src="images/bosspoverview.png" width="750" alt="Schematic overview of SSP-BO"/>
<br><em>SSP-BO conducts optimization in the space of vector-symbolic representations. An embedded candidate is decoded to an action, executed in the environment, and the reward is incorporated into a BLR posterior. The next candidate is found by optimizing the acquisition function in embedding space.</em>
</div>

---

## Features

- **Constant-time sample selection** — surrogate update and acquisition optimization costs are independent of *n*; up to 60–200× faster than GP-based BO.
- **Constant memory** — BLR parameter size is fixed at *O(d²)*, not *O(n²)*.
- **Structured and mixed spaces** — SSPs encode continuous, discrete, graph, trajectory, and mixed domains via a single algebraic framework, with no domain-specific modifications.
- **Neuromorphic compatibility** — acquisition optimization runs as continuous dynamics in embedding space, enabling deployment on Intel Loihi 2 and SpiNNaker with 30–188× energy reduction per sample.
- **Drop-in interface** — `BayesianOptimization` follows the [bayesian-optimization](https://github.com/bayesian-optimization/BayesianOptimization) API.

---

## How It Works

At each step the agent:
1. **Encodes** candidate locations as fixed-size SSP vectors `φ(x) ∈ ℝᵈ`.
2. **Updates** a BLR posterior over the embedded space via a rank-one update — *O(d²)*, no matrix inversion needed.
3. **Optimizes** the UCB acquisition function directly in embedding space via gradient ascent or neural dynamics:
   `θ* = argmax [ mᵀθ + λ √(β⁻¹ + θᵀΣθ) ]`
4. **Decodes** `θ*` back to an action: `x* = argmax φ(x)·θ*`

Because the BLR parameters `(m, Σ)` have fixed size *d* regardless of how many samples have been collected, steps 1–4 have cost that is entirely independent of *n* — the key to SSP-BO's scalability.

---

## Installation

Install with [uv](https://github.com/astral-sh/uv) (recommended) or pip:

```bash
uv pip install -e .
# or
pip install -e .
```

**Optional extras** install additional packages required to run specific experiment suites:

| Extra | Purpose |
|---|---|
| `basic` | matplotlib, pandas, torch, GPyTorch, BoTorch, Optuna — needed for GP/RFF baselines and plotting |
| `nengo` | Nengo SNN stack (Nengo ≥ 4, nengo-spa, nengo-ocl) — needed for neuromorphic experiments |
| `spinnaker` | SpiNNaker backend (requires Nengo < 3; **incompatible** with `nengo`) |
| `mcbo` | Mixed-variable / combinatorial tasks via MCBO |
| `nas` | NAS-Bench neural architecture search tasks |
| `dev` | pytest, mypy, jupyterlab |

```bash
uv pip install -e ".[basic]"      # standard continuous BO + baselines
uv pip install -e ".[nengo]"      # add Nengo SNN support
uv pip install -e ".[dev]"        # development tools
```

---

## Usage

```python
import numpy as np
import ssp_bayes_opt

def objective(x, y):
    return -(x**2 + y**2)

optimizer = ssp_bayes_opt.BayesianOptimization(
    f=objective,
    bounds=np.array([[-2, 2], [-2, 2]]),
    verbose=True,
    sampling_seed=0,
)

optimizer.maximize(
    init_points=5,
    n_iter=20,
    agent_type='ssp-hex',
    ssp_dim=151,
    length_scale=0.5,
    beta_ucb=10.0,
)

print(optimizer.max)
```

See [experiments/run_agent.py](experiments/run_agent.py) for a full example with multiple agent types and result saving, and [experiments/plot_1d_blr.py](experiments/plot_1d_blr.py) for a minimal 1-D demo.

---

## Paper experiments on YANA

Use the updated sibling `nengo-yana` and `yana` checkouts (deployment workspace,
cycle capture, typed capacity errors, and FIFO arbitration support are required).
Start the board runtime before running experiments.

```bash
# Inspect the full sweep without using hardware or creating artifacts.
python experiments/run_paper_experiments.py --yana-host kria.local --dry-run

# Four concurrent host workers, one shared FPGA.
python experiments/run_paper_experiments.py \
  --workspace-dir "$PWD" --yana-host kria.local --workers 4
```

The grid is the union of dense encoders at dimensions **25, 55, 121** and
50%/90% sparse encoders at dimension 55. Neurons per dimension are
`5,10,...,40` (25), `2,4,...,18` (55), and `1,...,8` (121). The 121 arm uses
explicitly 5 scales × 4 rotations: the old `--ssp-dim 120` default hex setup
silently produced 97 dimensions. Defaults are 100 optimization samples **plus
10 initial observations**, 5 trials, run-only power at 2 ms, and a 5 ns FPGA
clock period (200 MHz): **129 configurations / 645 trials** before hardware
exclusions. Change `--fpga-clock-period-ns` only to match the actual
bitstream; simulation timestep and host wall time are not FPGA latency.

For two systems, add `--num-shards 2 --shard-index 0` on one and
`--num-shards 2 --shard-index 1` on the other. Sharding is over the full grid,
independent of trial count or subset filters. Use `--functions`, `--ssp-dims`,
`--encoder-sparsities`, and `--num-neurons` to select a subset. Each worker
processes one configuration's trials sequentially; different configurations
prepare concurrently and arbitrate for the FPGA at sample boundaries.

Outputs live under `artifacts/paper/<function>/dim-..._sparsity-..._neurons-.../`:

- `config.json`: complete parameter identity, excluding requested trial count.
- `trial-NNNN/trial.json`: random seed, trial index, host, and start time.
- `trial-NNNN/trajectory/`: optimization `.npz`, including raw FPGA cycles and
  latencies, regret, objective values, and sample locations.
- `trial-NNNN/measurements/sample-NNNN.json`: raw run-only power samples/markers,
  seed, 1-based optimization sample, **0-based trajectory index**, execution
  window cycles, clock period, and summed on-chip latency.
- `trial-NNNN/run.log`, `command.json`, and validated/checksummed `complete.json`.

Re-run the same command to resume. Increasing `--num-trials` appends repetitions;
completed trials are validated and never overwritten. Incomplete attempts are
archived under `trial-NNNN-attempts/` and retried with their original randomly
chosen seed. A fully saved trial missing only its completion marker is recovered
without rerunning. Corrupt completed trials or mismatched configurations stop
execution rather than silently replacing results. Use a different workspace
when changing result-affecting settings such as `--num-samples`.

Every actual compiled network undergoes YANA resource validation. A typed
physical capacity failure records `failure.json` and `capacity-blocked.json`;
remaining trials of that configuration are **not run by policy**, not claimed
infeasible for all random seeds. Completed trials remain intact, failed seeds
are not replaced, and the blocked configuration is skipped on resume. Inspect
these records when analyzing results. Other errors stop execution and are not
misclassified as capacity exclusions.

Deployment files reuse one protected directory per worker under
`artifacts/paper/.work/shard-N/worker-N/deployment/fileset/`. Stale generated
files are removed before each sample; concurrent workers never share a fileset.
Other temporary files are isolated per worker and cleared before each trial.
Do not manually edit or delete scratch files while a worker owns them. Locks
prevent overlapping orchestrators for the same shard/configuration.

A small smoke run (use a separate workspace from paper results):

```bash
python experiments/run_paper_experiments.py \
  --workspace-dir /tmp/ssp-paper-smoke --yana-host kria.local --workers 2 \
  --functions himmelblau --ssp-dims 25 --encoder-sparsities 0 \
  --num-neurons 5 10 --num-trials 2 --num-samples 2
```

Repeat it to verify resume; then increase `--num-trials` to 3. `Ctrl-C` stops
active subprocesses and leaves attempts available for the next resume.

### Read-only plots while the sweep runs

```bash
.venv/bin/python experiments/plot_paper_experiments.py \
  --artifact-dirs "$PWD/artifacts/paper" \
  --output-dir /tmp/ssp-paper-plots
```

For separate shard trees, pass both locally accessible paths after
`--artifact-dirs` (for example a read-only mount of the other host's tree).
For one shared tree, pass it only once. The script does not fetch remote data.

Outputs are `plot1-dimensions.png`, `plot2-sparsity.png`, and `summary.csv`.
Each figure has separate panels for the three objective functions:

- Plot 1: dense encoders at 25, 55, and 121 dimensions.
- Plot 2: 55 dimensions at 50% and 90% encoder sparsity.
- Each point represents one configuration, averaged over its **currently
  completed** trials. Labels show neurons/dimension **(completed trial count)**.
  Lines connect increasing neuron counts within a dimension/sparsity group;
  they are not fitted Pareto frontiers. Use `--no-annotations` to hide labels.
- X is mean inference energy: for each optimization sample, multiply its mean
  run-only sensor power by its FPGA-reported latency, then average over samples
  and trials. The figure uses mJ; the CSV uses J. This is measured board/sensor
  power, not idle-subtracted or PL-only energy, and never host wall-clock time.
- Y is final cumulative average regret: mean regret across all observations
  (including the 10 initial observations), averaged over completed trials.

Only trials with an existing `complete.json` and matching parameter/file
checksums are read. In-progress trials, archived attempts, and invalid/missing
outputs are skipped without recovery or repair. Identical copies across roots
are counted once; conflicting copies or mixed fixed settings/budgets are
rejected. Empty panels say "No completed trials yet". Re-run to refresh the
snapshot as more trials finish.

The reader never takes sweep locks, writes completion markers, touches scratch
files, or contacts the FPGA. Outputs (including Matplotlib's temporary cache)
are kept outside **all** input trees; destinations inside an input tree are
rejected. Existing plot files are replaced atomically, not written through
symlinks/hardlinks to source files.

---

## Package Structure

```
ssp_bayes_opt/
├── bayesian_optimization.py        # Main BayesianOptimization class
├── nengo_bayesian_optimization.py  # Neuromorphic variant (Nengo backend)
├── blr.py                          # Bayesian linear regression surrogate
├── sspspace.py                     # SSP encoding: HexSSP, RandomSSP, etc.
├── network_solver.py               # Nengo network for acquisition optimization
├── agents/
│   ├── ssp_agent.py                # Core SSP-BO agent
│   ├── ssp_traj_agent.py           # Trajectory-space agent
│   ├── ssp_multi_agent.py          # Multi-agent variant
│   ├── ssp_nas_graph.py            # Graph-structured search (NAS-Bench)
│   ├── ssp_mcbo.py                 # Mixed-variable agent (MCBO tasks)
│   ├── gp_agent.py                 # GP baseline agent
│   └── rff_agent.py                # Random Fourier features baseline
└── util/
    └── discretized_function.py
```

---

## Citation

If you use SSP-BO in your research, please cite:

```bibtex
@article{furlong2025,
  author  = {P Michael Furlong and Nicole Dumont and Rika Antonova and Jeff Orchard and Chris Eliasmith},
  title   = {Compositional Neurosymbolic Representations Enable Efficient Active Exploration},
  journal = {Nature Communications},
  year    = {In press},
}
```
