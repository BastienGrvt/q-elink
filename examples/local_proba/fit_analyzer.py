import q_elink as ql
import numpy as np
import bastools as bst

suffix = "synth"
path_root = "examples/dataset/local_proba"
path_fit = path_root + "/fit/"
path_fig = path_root + "/fig/"

param_gauss = {}
param_plot = {}

# Set the analyzer and process the fit result
fit_analyzer = ql.FitResultAnalyzer()
fit_analyzer.load_data_raw(path_fit + f"fit_{suffix}_raw.json")
fit_analyzer.build_data()
fit_analyzer.save_data(path_fit + f"fit_{suffix}_analyzed.json")
stats = fit_analyzer.get_stat()


# Plot fitted model for local probability
def plot_fitted(path_plot_fitted, smooth_plot=False):
    fig_data, _ = fit_analyzer.plot_fit(param_gauss=param_gauss, n_sample=2000, smooth_plot=smooth_plot)
    fig_data.savefig(path_plot_fitted, dpi=300, bbox_inches='tight')
plot_fitted(f"{path_fig}/fig_fitted_{suffix}.png", smooth_plot=True)


# Plot fit shots
def plot_shots(path_plot_shots):
    fig_data, _ = fit_analyzer.plot_shots()
    fig_data.savefig(path_plot_shots, dpi=300, bbox_inches='tight')
plot_shots(f"{path_fig}/fig_shots_{suffix}.png")



# Plot histogram
def plot_histogram(path_plot_histogram):
    fig_histogram, _ = fit_analyzer.plot_histogram(param_plot=param_plot, param_gauss=param_gauss)
    fig_histogram.savefig(path_plot_histogram, dpi=300, bbox_inches='tight')
plot_histogram(path_fig + f"fig_fit_histogram_{suffix}.png")
