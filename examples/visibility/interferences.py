import q_elink as ql
import numpy as np
import matplotlib.pyplot as plt

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

# Sweep the analysis phase and compute the interference fringes
phi = np.linspace(-np.pi, np.pi, 200)
proba_coinc = [model_visib.get_proba_coincidence(p) for p in phi]
proba_bell = [model_visib.get_proba_coincidence_pert(p) for p in phi]

# Plot the fringes against the perturbative Bell-pair reference
fig, ax = plt.subplots()
ax.plot(phi, proba_coinc, label="Elementary link")
ax.plot(phi, proba_bell, "k--", label="Bell pair (perturbative)")
ax.set_xlabel(r"Analysis phase $\phi$")
ax.set_ylabel("Coincidence probability")
ax.grid(True)
ax.legend()

print(f"Visibility: {model_visib.get_visibility()}")
plt.show()
