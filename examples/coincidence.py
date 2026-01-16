import q_elink as ql


dc = 1e-4

param_dict = {
        "elink": {  
                  "p_A": 0.1,
                  "p_B": 0.3,
                  "eta_0": 1e-2,
                  "eta_A": 0.6,
                  "eta_B": 0.6,
                  "dc_0": dc,
                  "dc_A": dc,
                  "dc_B": dc,
                  },
        "coinc": { 
                  "eta_T": 0.8,
                  "eta_J": 0.2,
                  "dc_T": 0,
                  "dc_J": 0,
                  },
        }


elink = ql.ElementaryLink()
elink.set_param(param_dict["elink"])

local_proba_coinc = ql.CoincidenceProbabilityModel(elink)
local_proba_coinc.set_param(param_dict["coinc"])

side = 'A'
print(f"proba_coinc = {local_proba_coinc.get_proba(side)}")
