"""Fit the local-probability model on a synthetic dataset.

Loads the dataset produced by examples/dataset/create_dataset_local_proba.py,
runs the Monte-Carlo fit, and stores the raw fit result under local/dataset.
Figures go to local/fig.
"""
import json
from pathlib import Path

import q_elink as ql

# ----- Paths -----
suffix = "synth"
path_dataset = Path("local/dataset")
path_fig = Path("local/fig")
path_dataset.mkdir(parents=True, exist_ok=True)
path_fig.mkdir(parents=True, exist_ok=True)

# ----- Load the synthetic dataset -----
with open(path_dataset / "local_proba.json") as f:
    dataset = json.load(f)

local_proba_exp = ql.LocalProbabilityExperiment()
local_proba_exp.set_pump(dataset["pump_dict"])
local_proba_exp.set_proba(dataset["proba_dict"])
fig, axs = local_proba_exp.plot_data()
fig.savefig(path_fig / f"fig_pij_{suffix}.png", dpi=300, bbox_inches="tight")

# ----- Fit parameters -----
n_fit = 50
pij_pond = [1, 1, 1, 0.1]
relative = True
init_value = {
    "eta_0": [1e-3, 0.1],
    "eta_A": [0.05, 0.15],
    "eta_B": [0.05, 0.15],
    "dc_0": [-5, -2],
    "dc_A": [-5, -3],
    "dc_B": [-5, -3],
}

# ----- Fit -----
fit = ql.LocalProbabilityFitter()
fit.set_data(local_proba_exp)
fit.set_init_value(init_value)
fit.set_fit(n_fit, pij_pond=pij_pond, relative=relative)
result_fit = fit.fit()
fit.save_result(path_dataset / f"fit_{suffix}_raw.json")
