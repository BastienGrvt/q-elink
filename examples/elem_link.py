import q_elink as ql

dc = 1e-4

param_elink = {
    'p_A': 0.01,
    'p_B': 0.08,
    'eta_0': 1e-3,
    'eta_A': 0.01,
    'eta_B': 0.01,
    'dc_0': dc,
    'dc_A': dc,
    'dc_B': dc,
}


param_detect = {
    'eta_A': 1,
    'eta_B': 1,
    'dc_A': 0,
    'dc_B': 0,
        }

alpha = 0.5
beta = -0.5

elink = ql.ElemLink()
elink.set_param(param_elink)

local_proba = ql.LocalProbaModel(elink)
local_proba.set_param(param_detect)

proba_herald = local_proba.get_p_herald()
p00, p01, p10, p11 = local_proba.get_proba(alpha=alpha, beta=beta)
fig, ax = local_proba.plot_proba()



# Print the results
print("=== Probabilities ===")
print(f"Heralding probability: {proba_herald}")
print(f"Local probabilities:")
print(f"p00 = {p00}")
print(f"p01 = {p01}")
print(f"p10 = {p10}")
print(f"p11 = {p11}")
