
import q_elink as ql


# Default parameters
p_A, p_B = 1e-5, 1e-5
dc = 0
eta_T = 1e-3
eta_D = 0.9
eta_S = 0.5
eta_disp = 0.8
eta_coinc = 0.8

# Displacement
alpha, beta = 0.5, -0.5

# Dirctionnary for param setter
param_dict = {
        "herald": {
            "eta_0": eta_T,
            "dc_0": dc,
            },
        "state": {
            "eta_A": eta_S,
            "eta_B": eta_S,
            "dc_A": dc,
            "dc_B": dc,
            },
        "detectors": {
            "eta": eta_D,
            "dc": 0,
            },
        "displacement": {
            "eta_A": eta_disp,
            "eta_B": eta_disp,
            "dc_A": 0,
            "dc_B": 0, 
            },
        "coincidence": {
            "alice": {
                "eta_T": eta_coinc,
                "eta_J": eta_coinc,
                "dc_T": 0,
                "dc_J": 0,
                },
            "bob": {
                "eta_T": eta_coinc,
                "eta_J": eta_coinc,
                "dc_T": 0,
                "dc_J": 0,
                },
            },
        }


ent_wit = ql.EntWitness()
ent_wit.set_param(param_dict)
ent_wit.set_pump(p_A, p_B)


w_mean = ent_wit.get_w_mean(alpha, beta)
w_ppt = ent_wit.get_w_ppt(alpha, beta)
witness = ent_wit.get_witness(alpha, beta)


print(f"w_mean = {w_mean}")
print(f"w_ppt = {w_ppt}")
print(f"w_witness = {witness}")






