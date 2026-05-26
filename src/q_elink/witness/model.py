from ._imports import *
from matplotlib import colors, cm


class EntanglementWitness():
    # TODO
    def __init__(self):
        self.elink = ElementaryLink()
        self.local_proba_rho = LocalProbabilityModel(self.elink)
        self.local_proba_obs = LocalProbabilityModel(self.elink)
        self.local_proba_coinc_A = CoincidenceProbabilityModel(self.elink)
        self.local_proba_coinc_B = CoincidenceProbabilityModel(self.elink)

        self.param_dict: dict = {
                "elink": {
                    "eta_0": None,
                    "dc_0": None,
                    },
                "w_rho": {
                    "eta_A": None,
                    "eta_B": None,
                    "dc_A": None,
                    "dc_B": None,
                    },
                "w_obs": {
                    "eta_A": None,
                    "eta_B": None,
                    "dc_A": None,
                    "dc_B": None,
                    },
                "coef": {
                    "eta_A": None,
                    "eta_B": None,
                    "dc_A": None,
                    "dc_B": None,
                    },
                "coinc": {
                    "alice": {
                        "eta_T": None,
                        "eta_J": None,
                        "dc_T": None,
                        "dc_J": None,
                        },
                    "bob": {
                        "eta_T": None,
                        "eta_J": None,
                        "dc_T": None,
                        "dc_J": None,
                        },
                    },
                }

        self.input_param: dict = {
                "herald": {
                    "eta_0": None,
                    "dc_0": None
                    },
                "state": {
                    "eta_A": None,
                    "eta_B": None,
                    "dc_A": None,
                    "dc_B": None 
                    },
                "detectors": {
                    "eta": None,
                    "dc": None 
                    },
                "displacement": {
                    "eta_A": None,
                    "eta_B": None,
                    "dc_A": None,
                    "dc_B": None 
                    },
                "coincidence": {
                    "alice": {
                        "eta_T": None,
                        "eta_J": None,
                        "dc_T": None,
                        "dc_J": None,
                        },
                    "bob": {
                        "eta_T": None,
                        "eta_J": None,
                        "dc_T": None,
                        "dc_J": None,
                        },
                    },
                }


    def check_integrity(self):
        # def check_param(param_dict):
        #     for key, val in param_dict.items():
        #         if isinstance(val, dict):
        #             return check_param(val)
        #         else:
        #             if val is None:
        #                 raise ValueError(f"Parameter {key} have not been set.") 
        # if self.elink is None or self.local_proba_rho is None:
        #     raise ValueError("Please set the model with `set_model()` first.")
        # check_param(self.param_dict)
        pass


    def set_param(self, new_param): 

        def recursive_update(default_dict, new_val):
                for key, val in new_val.items():
                    if isinstance(val, dict):
                        recursive_update(default_dict[key], val)
                    else:
                        default_dict[key] = val

        recursive_update(self.input_param, new_param)

    def get_param(self):
        return self.real_param_dict

    def set_pump(self, p_A, p_B):
        self.elink.p_A = p_A
        self.elink.p_B = p_B

    def _build_raw_param(self):
        self.check_integrity()
        self.param_dict: dict = {
                "elink": {
                    "eta_0": self.input_param["herald"]["eta_0"],
                    "dc_0": self.input_param["herald"]["dc_0"],
                    "eta_A": self.input_param["state"]["eta_A"],
                    "eta_B": self.input_param["state"]["eta_B"],
                    "dc_A": self.input_param["state"]["dc_A"],
                    "dc_B": self.input_param["state"]["dc_B"],
                    },
                "proba_rho": {
                    "eta_A": self.input_param["detectors"]["eta"],
                    "eta_B": self.input_param["detectors"]["eta"],
                    "dc_A": self.input_param["detectors"]["dc"],
                    "dc_B": self.input_param["detectors"]["dc"],
                    },
                "proba_obs": {
                    "eta_A": self.input_param["displacement"]["eta_A"] * self.input_param["detectors"]["eta"],
                    "eta_B": self.input_param["displacement"]["eta_B"] * self.input_param["detectors"]["eta"],
                    "dc_A": self.input_param["displacement"]["dc_A"] + self.input_param["detectors"]["dc"],
                    "dc_B": self.input_param["displacement"]["dc_B"] + self.input_param["detectors"]["dc"],
                    },
                "coef": {
                    "eta_A": self.input_param["displacement"]["eta_A"],
                    "eta_B": self.input_param["displacement"]["eta_B"],
                    "dc_A": self.input_param["displacement"]["dc_A"],
                    "dc_B": self.input_param["displacement"]["dc_B"],
                    },
                "coinc": {
                    "alice": {
                        "eta_T": self.input_param["coincidence"]["alice"]["eta_T"] * self.input_param["detectors"]["eta"],
                        "eta_J": self.input_param["coincidence"]["alice"]["eta_J"] * self.input_param["detectors"]["eta"],
                        "dc_T": self.input_param["coincidence"]["alice"]["dc_T"] + self.input_param["detectors"]["dc"],
                        "dc_J": self.input_param["coincidence"]["alice"]["dc_J"] + self.input_param["detectors"]["dc"],
                        },
                    "bob": {
                        "eta_T": self.input_param["coincidence"]["bob"]["eta_T"] * self.input_param["detectors"]["eta"],
                        "eta_J": self.input_param["coincidence"]["bob"]["eta_J"] * self.input_param["detectors"]["eta"],
                        "dc_T": self.input_param["coincidence"]["bob"]["dc_T"] + self.input_param["detectors"]["dc"],
                        "dc_J": self.input_param["coincidence"]["bob"]["dc_J"] + self.input_param["detectors"]["dc"],
                        },
                    },
                }

    def _build_real_param(self):
        pass

        
    def _get_not_none(self, arg, arg_type=object):
        if arg is not None:
            if isinstance(arg, arg_type):
                return arg
            else:
                raise TypeError(f"TypeError in the parameters dictionary, {arg_type} expected.")
        else:
            raise ValueError("Please set the parameters with `set_parameter()` first.")

    def _set_not_none(self, class_parent, **kwargs):
        if class_parent is not None:
            for name, val in kwargs.items():
                if val is not None:
                    setattr(class_parent, name, val) 
        else:
            raise ValueError(f"Class {class_parent} not knows.")

    def _coef_0(self, alpha, beta, eta_A, eta_B): # w_00
        factor_1 = 2 * m.exp(-eta_A * np.abs(alpha)**2) - 1
        factor_2 = 2 * m.exp(-eta_B * np.abs(beta)**2) - 1
        return factor_1 * factor_2 

    def _coef_1(self, alpha, beta, eta_A, eta_B): # w_01
        factor_1 = 2 * m.exp(-eta_A * np.abs(alpha)**2) - 1
        factor_2 = 2 * (1 + eta_B * (eta_B * np.abs(beta)**2 - 1)) * m.exp(-eta_B * np.abs(beta)**2) - 1
        return factor_1 * factor_2 

    def _coef_2(self, alpha, beta, eta_A, eta_B): # w_10
        factor_1 = 2 * (1 + eta_A * (eta_A * np.abs(alpha)**2 - 1)) * m.exp(-eta_A * np.abs(alpha)**2) - 1
        factor_2 = 2 * m.exp(-eta_B * np.abs(beta)**2) - 1
        return factor_1 * factor_2

    def _coef_3(self, alpha, beta, eta_A, eta_B): # w_11
        factor_1 = 2 * (1 + eta_A * (eta_A * np.abs(alpha)**2 - 1)) * m.exp(-eta_A * np.abs(alpha)**2) - 1
        factor_2 = 2 * (1 + eta_B * (eta_B * np.abs(beta)**2 - 1)) * m.exp(-eta_B * np.abs(beta)**2) - 1
        return factor_1 * factor_2

    def _coef_4(self, alpha, beta, eta_A, eta_B): # w_c01 + w_c10
        return 8 * eta_A*eta_B * (alpha * np.conjugate(beta)).real * np.exp(-(eta_A*np.abs(alpha)**2 + eta_B*np.abs(beta)**2))

    def _get_coef(self, alpha, beta, eta_A, eta_B):
        return self._coef_0(alpha, beta, eta_A, eta_B), self._coef_1(alpha, beta, eta_A, eta_B), self._coef_2(alpha, beta, eta_A, eta_B), self._coef_3(alpha, beta, eta_A, eta_B), self._coef_4(alpha, beta, eta_A, eta_B)

    def _init_witness(self):
        self._build_raw_param()
        elink_param = self.param_dict["elink"]
        self.elink.set_param(elink_param)
        proba_rho_param = self.param_dict["proba_rho"]
        self.local_proba_rho.set_param(proba_rho_param)
        proba_obs_param = self.param_dict["proba_obs"]
        self.local_proba_obs.set_param(proba_obs_param)
        proba_coinc_param = self.param_dict["coinc"]
        self.local_proba_coinc_A.set_param(proba_coinc_param["alice"])
        self.local_proba_coinc_B.set_param(proba_coinc_param["bob"])

    def get_w_ppt(self, alpha, beta):
        self.check_integrity()
        self._init_witness()
        # Get the displaced local probabities (from the W observable)
        p_ij = self.local_proba_rho.get_proba()
        p00, p01, p10, p11 = p_ij
        # Get the 2-photons boundaries
        p_bound_A = 2*self.local_proba_coinc_A.get_p11('A')
        p_bound_B = 2*self.local_proba_coinc_B.get_p11('B')
        # print(p_bound_A)
        # print(p_bound_B)
        p_bound_AB = p_bound_A + p_bound_B
        # Get the coefficientefficiency
        eta_wppt_A = self.param_dict["coef"]["eta_A"]
        eta_wppt_B = self.param_dict["coef"]["eta_B"]
        coefs = self._get_coef(alpha, beta, eta_wppt_A, eta_wppt_B)
        coef_0, coef_1, coef_2, coef_3, coef_4 = coefs
        # In qubit space:
        terms_in = np.zeros(5)
        terms_in[0] = coef_0 * p00
        terms_in[1] = max(coef_1 * (p01-p_bound_A), coef_1 * p01)
        terms_in[2] = max(coef_2 * (p10-p_bound_B), coef_2 * p10)
        terms_in[3] = coef_3 * p11 #max(coef_3 * (p11-p_bound_AB), coef_3 * (p11-p_bound_AB)*p11)
        terms_in[4] = coef_4 * np.sqrt(p00*p11) # max(coef_4 * np.sqrt(p00*p11), coef_4 * np.sqrt(p00*(p11-p_bound_AB)))
        # print(" ----- New Point -----")
        # if p11 < 0:
        #     print(self.input_param)
        #     print(p00, p01, p01, p11)
        # Out of qubit space:
        terms_out = np.zeros(2)
        p_bound = p_bound_A + p_bound_B
        lambd = lambda x, eta: 2*(1-eta)+ (2-eta**2) * np.abs(x)**2
        b_max = 2 * eta_wppt_A * eta_wppt_B * np.sqrt(2* np.abs(lambd(beta, eta_wppt_B))**2 + 2*np.abs(lambd(alpha, eta_wppt_A))**2 ) * np.exp(-eta_wppt_A*np.abs(alpha)**2-eta_wppt_B*np.abs(beta)**2)
        terms_out[0] = p_bound
        terms_out[1] = 2*b_max*np.sqrt(abs(p_bound*(1-p_bound)))
        return sum(terms_in) + sum(terms_out)


    def get_w_mean(self, alpha, beta):
        # Set the parameters
        self.check_integrity()
        self._init_witness()
        coef_alpha = 1 / np.sqrt( self.param_dict["elink"]["eta_A"] * self.param_dict["proba_rho"]["eta_A"])
        coef_beta = 1 / np.sqrt( self.param_dict["elink"]["eta_B"] * self.param_dict["proba_rho"]["eta_B"])
        p00, p01, p10, p11 = self.local_proba_obs.get_proba(coef_alpha * alpha, coef_beta * beta)
        return p00 - p01 - p10 + p11


    def get_witness(self, alpha, beta):
        w_mean = self.get_w_mean(alpha, beta)
        w_ppt = self.get_w_ppt(alpha, beta)
        return w_ppt - w_mean


    def plot_cmap(self, x_bound, y_bound, N=30, N_lvl=300, mask=False):
        self.check_integrity()
        x_min, x_max = x_bound
        y_min, y_max = y_bound
        x = np.linspace(x_min, x_max, N)
        y = np.linspace(y_min, y_max, N)
        X, Y = np.meshgrid(x, y)
        Z = np.zeros((N, N))
        for i, _ in enumerate(tqdm(X)):
            for j, _ in enumerate(Y):
                param_x = X[i, j]
                param_y = Y[i, j]
                Z[i, j] = self.get_witness(param_x, param_y)
        z_min, z_max = (np.min(Z), np.max(Z))
        # print(f"z_min = {z_min} \nz_max = {z_max}")
        z_min, z_max = (min(z_min, 0), 0) if mask else (z_min, z_max)
        
        cmap = plt.get_cmap('viridis_r')
        normalizer = colors.Normalize(vmin=z_min, vmax=z_max)
        fig, ax = plt.subplots(1, layout="constrained")
        cs = ax.contourf(X, Y, Z, levels=N_lvl, cmap=cmap, norm=normalizer, antialiased=True)
        # cs = ax.pcolormesh(X, Y, Z, cmap=cmap, norm=normalizer,shading='gouraud')
        ax.set_xlabel(r'Alpha', fontsize=12)
        ax.set_ylabel(r'Beta', fontsize=12)
        ax.grid()
        ccf = cm.ScalarMappable(norm=normalizer, cmap=cmap)
        cbar = fig.colorbar(ccf, ax=ax)
        cbar.set_label('$w_{ppt}-<\mathcal{O}>$')
        return fig

