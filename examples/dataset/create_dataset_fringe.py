"""Generate a synthetic interference-fringe dataset to test a visibility fit.

Sets a known elementary link, computes the coincidence probability as a
function of the analysis phase theta over [-pi, pi], then adds noise: a
constant vertical offset, a small global phase drift, and per-point scatter.
The dataset is stored in local/dataset/.
"""
import json
from pathlib import Path

import numpy as np

import q_elink as ql

# ----- Output path -----
path_dataset = Path("local/dataset")
path_dataset.mkdir(parents=True, exist_ok=True)
out_file = path_dataset / "fringe.json"

# ----- Ground-truth model -----
rng_seed = 42
rng = np.random.default_rng(rng_seed)

ground_truth = {
    "p_A": 0.05,
    "p_B": 0.05,
    "eta_0": 1e-2,
    "eta_A": 0.1,
    "eta_B": 0.1,
    "dc_0": 1e-4,
    "dc_A": 1e-4,
    "dc_B": 1e-4,
}

elink = ql.ElementaryLink()
elink.set_param(ground_truth)
model_visib = ql.InterferenceModel(elink)

# ----- Phase sampling -----
n_data = 10
theta = np.linspace(-np.pi, np.pi, n_data)

# ----- Noise model -----
offset = 5e-3          # constant vertical offset
phase_drift = 0.15     # small global phase drift (rad)
noise_std_ratio = 0.03  # per-point relative scatter

proba_clean = np.array([model_visib.get_proba_coincidence(t + phase_drift) for t in theta])
proba_noisy = proba_clean + offset + rng.normal(0, noise_std_ratio * np.abs(proba_clean))

# ----- Save -----
dataset = {
    "ground_truth": ground_truth,
    "noise": {
        "offset": offset,
        "phase_drift": phase_drift,
        "noise_std_ratio": noise_std_ratio,
    },
    "theta": theta.tolist(),
    "proba_coincidence": proba_noisy.tolist(),
}
with open(out_file, "w") as f:
    json.dump(dataset, f, indent=4)

print(f"Fringe dataset written to {out_file}")
