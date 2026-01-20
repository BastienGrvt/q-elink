import q_elink as ql


# Set the model parameters
side = 'A'  # Alice's side
param_dict = {
        "elink": {  # Elementary link
                  "p_A": 0.1,
                  "p_B": 0.3,
                  "eta_0": 1e-2,
                  "eta_A": 0.6,
                  "eta_B": 0.6,
                  "dc_0": 1e-4,
                  "dc_A": 1e-4,
                  "dc_B": 1e-4,
                  },
        "coinc": {  # Coincidence set-up
                  "eta_T": 0.8,
                  "eta_J": 0.2,
                  "dc_T": 0,
                  "dc_J": 0,
                  },
        }


# Set the elementary link model
elink = ql.ElementaryLink()
elink.set_param(param_dict["elink"])

# Set the coincidence model
coinc_proba = ql.CoincidenceProbabilityModel(elink)
coinc_proba.set_param(param_dict["coinc"])

# Show the status
coinc_proba.show()

# Compute the coincidence probability at Alice's side
print(f"proba_coinc = {coinc_proba.get_proba(side)}")
