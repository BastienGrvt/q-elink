import q_elink as ql

# Elementary link parameters
param_elink = {
    "p_A": 0.05,
    "p_B": 0.05,
    "eta_0": 1e-2,
    "eta_A": 0.1,
    "eta_B": 0.1,
    "dc_0": 1e-4,
    "dc_A": 1e-4,
    "dc_B": 1e-4,
}

# Create the elementary link and the interference model
elink = ql.ElementaryLink()
elink.set_param(param_elink)
model_visib = ql.InterferenceModel(elink)

# Coincidence probability at zero phase and the resulting fringe visibility
proba_coinc = model_visib.get_proba_coincidence(phi=0.0)
visibility = model_visib.get_visibility()

print("=== Interference ===")
print(f"Coincidence probability (phi=0): {proba_coinc}")
print(f"Visibility:                      {visibility}")
