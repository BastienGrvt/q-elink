"""Analyze and plot the local-probability fit result.

Reads the raw fit result written by fit_proba.py from local/dataset, builds
the processed statistics, and saves the analysis plots under local/fig.
"""
from pathlib import Path

import q_elink as ql

suffix = "synth"
path_dataset = Path("local/dataset")
path_fig = Path("local/fig")
path_dataset.mkdir(parents=True, exist_ok=True)
path_fig.mkdir(parents=True, exist_ok=True)

param_gauss = {}
param_plot = {}

# Set the analyzer and process the fit result
fit_analyzer = ql.FitResultAnalyzer()
fit_analyzer.load_data_raw(path_dataset / f"fit_{suffix}_raw.json")
fit_analyzer.build_data()
fit_analyzer.save_data(path_dataset / f"fit_{suffix}_analyzed.json")
stats = fit_analyzer.get_fitted_param()


# Plot fitted model for local probability
def plot_fitted(path_plot_fitted, smooth_plot=False):
    fig_data, _ = fit_analyzer.plot_fit(param_gauss=param_gauss, n_sample=2000, smooth_plot=smooth_plot)
    fig_data.savefig(path_plot_fitted, dpi=300, bbox_inches="tight")
plot_fitted(path_fig / f"fig_fitted_{suffix}.png", smooth_plot=True)


# Plot fit shots
def plot_shots(path_plot_shots):
    fig_data, _ = fit_analyzer.plot_shots()
    fig_data.savefig(path_plot_shots, dpi=300, bbox_inches="tight")
plot_shots(path_fig / f"fig_shots_{suffix}.png")


# Plot histogram
def plot_histogram(path_plot_histogram):
    fig_histogram, _ = fit_analyzer.plot_histogram(param_plot=param_plot, param_gauss=param_gauss)
    fig_histogram.savefig(path_plot_histogram, dpi=300, bbox_inches="tight")
plot_histogram(path_fig / f"fig_fit_histogram_{suffix}.png")


