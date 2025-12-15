import q_elink as ql
import numpy as np


# Noise function for data generation

# Path for data
path = "examples/tmp/"

# Fake data parameters
n_data = 5
default_param = {
    "eta_0" : 1e-2,
    "eta_A" : 0.1,
    "eta_B" : 0.1,
    "dc_0" : 1e-4,
    "dc_A" : 1e-4,
    "dc_B" : 1e-4,
}

# Set fit param
pij_pond = [1, 1, 1, 0.1]
relative = True
init_value = {
    "eta_0": [1e-3, 0.1],
    "eta_A": [0.05, 0.15],
    "eta_B": [0.05, 0.15],
    "dc_0": [1e-4, 1e-2],
    "dc_A": [1e-5, 1e-3],
    "dc_B": [1e-5, 1e-3],
}

# Set fake data
elink = ql.ElemLink()
elink_model = ql.LocalProbaModel(elink) 
elink.set_param(default_param)

def noise(x, std_ratio=0.1):
    return x + np.random.normal(0, std_ratio * np.abs(x))

def wrapper(x, y, foo):
    elink_model.set_pump(p_A=x, p_B=y)
    return foo()

p_A = np.random.uniform(low=0.01, high=0.15, size=n_data) 
p_B = noise(p_A, std_ratio=0.03) 

pump_data = {
    "p_A": p_A,
    "p_B": p_B,
}

proba_data = {
        "p00": np.vectorize(lambda x, y: wrapper(x, y, elink_model.get_p00))(p_A, p_B),
        "p01": np.vectorize(lambda x, y: wrapper(x, y, elink_model.get_p01))(p_A, p_B),
        "p10": np.vectorize(lambda x, y: wrapper(x, y, elink_model.get_p10))(p_A, p_B),
        "p11": np.vectorize(lambda x, y: wrapper(x, y, elink_model.get_p11))(p_A, p_B),
}

for key, value in proba_data.items():
    proba_data[key] = noise(value, std_ratio = 0)


# Set generated experimental point as dataset
local_proba_exp = ql.LocalProbaExperiment()
local_proba_exp.set_proba(proba_data)
local_proba_exp.set_pump(pump_data)
fig = local_proba_exp.plot_data()
fig.savefig(f"{path}/pij_exp.png", dpi=300, bbox_inches='tight')


# Do the fit
fit = ql.FitLocalProba()
fit.set_data(local_proba_exp)
fit.set_init_value(init_value)
fit.set_fit(10, pij_pond=pij_pond, relative=relative)
result_fit = fit.fit()
fit.save_result(f'{path}/fit_test.json')


# Process the data and show
data_process = ql.FitDataProcess()
data_fit = data_process.load_data(f'{path}/fit_test.json')
processed_data = data_process.build_data(data_fit)
data_process.save_data(processed_data, f'{path}/processed_fit.json')
fig = data_process.plot_data(processed_data)
fig.savefig(f"{path}/fit_test.png", dpi=300, bbox_inches='tight')
