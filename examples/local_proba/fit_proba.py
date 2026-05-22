import q_elink as ql
import numpy as np
import os


# Path for data
suffix = "synth"
path_root = "examples/dataset/local_proba/"
path_fit = f"{path_root}/fit/"
path_fit = f"{path_root}/fig/"


# Set fit param
n_fit = 200
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

# Parameters of the model for synthetic experemental data
rng_seed = 42
n_data = 5
p_min, p_max = 0.03, 0.15
default_param = {
    "eta_0" : 1e-2,
    "eta_A" : 0.1,
    "eta_B" : 0.1,
    "dc_0" : -4,
    "dc_A" : -4,
    "dc_B" : -4,
}


# ----- Set synthetic experimental data -----


# Noise function for synthetic data generation
rng = np.random.default_rng(rng_seed)
def noise(x, std_ratio=0.1):
    return x + rng.normal(0, std_ratio * np.abs(x))

def wrapper(x, y, foo):
    elink_synth.set_pump(p_A=x, p_B=y)
    return foo()

# Set class models synthetic data generation
elink_synth = ql.ElementaryLink()
local_proba_synth = ql.LocalProbabilityModel(elink_synth) 
elink_synth.set_param(default_param, log_dc=True)

# Pump parameters
p_A = np.linspace(p_min, p_max, n_data) 
p_B = noise(p_A, std_ratio=0.03) 

data_pump_synth = {
    "p_A": p_A,
    "p_B": p_B,
}

# Generate synthetic exprerimental data
data_proba_synth = {
        "p00": np.vectorize(lambda x, y: wrapper(x, y, local_proba_synth.get_p00))(p_A, p_B),
        "p01": np.vectorize(lambda x, y: wrapper(x, y, local_proba_synth.get_p01))(p_A, p_B),
        "p10": np.vectorize(lambda x, y: wrapper(x, y, local_proba_synth.get_p10))(p_A, p_B),
        "p11": np.vectorize(lambda x, y: wrapper(x, y, local_proba_synth.get_p11))(p_A, p_B),
}

data_proba_synth_noisy = {}
for key, value in data_proba_synth.items():
    data_proba_synth_noisy[key] = noise(value, std_ratio = 0.01)


# ----- Fit -----
# Set generated experimental point as dataset
local_proba_exp = ql.LocalProbabilityExperiment()
local_proba_exp.set_proba(data_proba_synth_noisy)
local_proba_exp.set_pump(data_pump_synth)
fig, axs = local_proba_exp.plot_data()
fig.savefig(f"{path_ig}/fig_pij_{suffix}.png", dpi=300, bbox_inches='tight')


# Fit the data
fit = ql.LocalProbabilityFitter()
fit.set_data(local_proba_exp)
fit.set_init_value(init_value)
fit.set_fit(n_fit, pij_pond=pij_pond, relative=relative)
result_fit = fit.fit()
fit.save_result(f'{path_fit}/fit_{suffix}_raw.json')
