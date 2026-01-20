import q_elink as ql

# Elementary link parameters
param_elink = {
    'p_A': 0.01,
    'p_B': 0.08,
    'eta_0': 1e-3,
    'eta_A': 0.01,
    'eta_B': 0.01,
    'dc_0': 1e-4,
    'dc_A': 1e-4,
    'dc_B': 1e-4,
}



# Create an instance of the ElementaryLink class
elink = ql.ElementaryLink()
elink.set_param(param_elink)

# Show the link status
elink.show()
