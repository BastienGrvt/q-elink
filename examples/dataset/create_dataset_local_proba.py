"""Generate a synthetic local-probability dataset to test the fit.

Sets a known elementary link, computes the local probabilities
(p00, p01, p10, p11) without displacement (alpha = beta = 0) for a handful of
pump values, adds a bit of noise, and stores the result in local/dataset/.
The dataset is consumed by examples/local_proba/fit_proba.py.
"""
import json
from pathlib import Path

import numpy as np

import q_elink as ql

# ----- Output path -----
path_dataset = Path("local/dataset")
path_dataset.mkdir(parents=True, exist_ok=True)
out_file = path_dataset / "local_proba.json"

# ----- Ground-truth model -----
rng_seed = 42
rng = np.random.default_rng(rng_seed)

# NB: dark counts are given on a log10 scale (set_param with log_dc=True).
ground_truth = {
    "eta_0": 1e-2,
    "eta_A": 0.1,
    "eta_B": 0.1,
    "dc_0": -4,
    "dc_A": -4,
    "dc_B": -4,
}

elink = ql.ElementaryLink()
elink.set_param(ground_truth, log_dc=True)
local_proba = ql.LocalProbabilityModel(elink)

# ----- Pump sampling (max pump p = 0.2) -----
n_data = 5
p_min, p_max = 0.03, 0.2
p_A = np.linspace(p_min, p_max, n_data)
p_B = p_A + rng.normal(0, 0.03 * np.abs(p_A))  # slight A/B asymmetry

# ----- Compute local probabilities (no displacement: alpha = beta = 0) -----
def sample(getter, p_a, p_b):
    local_proba.set_pump(p_A=p_a, p_B=p_b)
    return getter()

proba_dict = {}
for key, getter in [
    ("p00", local_proba.get_p00),
    ("p01", local_proba.get_p01),
    ("p10", local_proba.get_p10),
    ("p11", local_proba.get_p11),
]:
    values = np.array([sample(getter, p_a, p_b) for p_a, p_b in zip(p_A, p_B)])
    proba_dict[key] = values + rng.normal(0, 0.01 * np.abs(values))  # relative noise

# ----- Save -----
dataset = {
    "ground_truth": ground_truth,
    "pump_dict": {"p_A": p_A.tolist(), "p_B": p_B.tolist()},
    "proba_dict": {key: value.tolist() for key, value in proba_dict.items()},
}
with open(out_file, "w") as f:
    json.dump(dataset, f, indent=4)

print(f"Local-probability dataset written to {out_file}")
