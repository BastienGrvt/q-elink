from ._imports import *

from .elem_link_model import ElemLink, LocalProbaModel

from matplotlib import cm, colors

class LocalProbaCoincidence():
    def __init__(self, elink: ElemLink):
        self.eta_T = None
        self.eta_J = None
        self.dc_T = None
        self.dc_J = None
        self.side = None
        self.elink = elink
        self.local_proba_model = LocalProbaModel(self.elink)


    def set_param(self, param_dict):
        for key in ['eta_T', 'eta_J', 'dc_T', 'dc_J']:
            if key in param_dict:
                setattr(self, key, param_dict[key])
            else:
                warnings.warn(f"Uknown parameter: {key}.", UserWarning)

    def check_integrity(self):
        """Check if the parameters have been set before any calculation.

        Raises:
            ValueError: If any required parameter is not set.
        """
        eff_param = {
            'eta_T': self.eta_T,
            'eta_J': self.eta_J,
            'dc_T': self.dc_T,
            'dc_J': self.dc_J,
        }
        model = {
            'local_proba_model': self.local_proba_model,
            'elink': self.local_proba_model.elink,
        }
        # Check the models
        # for name, value in model.items():
        #     if value is None:
        #         raise ValueError(f"Model not set (`{name} = None`, please set the model first.")
        # Check for parameters
        for name, value in eff_param.items():
            if value is None:
                raise ValueError(f"Parameter {name} has not been set, please set {name} first.")
        # if self.side is None:
        #     raise ValueError(f"Please set the side via `set_side()` first.")
        # elif self.side not in ['A', 'B']:
        #     raise ValueError(f"The side choice `side` must be `A` or `B` for Alice or Bob.")

    def _set_not_none(self, class_parent, **kwargs):
        for name, val in kwargs.items():
            if val is not None:
                setattr(class_parent, name, val) 

    def _get_svd(self,p1, p2, eta_0, dc_0, R_T, R_J, dc_T, dc_J, cond_0):
        R_01, R_02 = 1 - cond_0[0] * eta_0, 1 - cond_0[1] * eta_0
        mat_svd_square = (1/4)*np.array([
                            [ p1*R_01*R_T, p1*R_02*R_T],
                            [ p1*R_01*R_J, p1*R_02*R_J],
                            [ 2*p2*R_01,   -2*p2*R_02]])
        mat_svd = np.sqrt(np.abs(mat_svd_square)) * np.sign(mat_svd_square)
        s = np.linalg.svd(mat_svd, full_matrices=True, compute_uv=False, hermitian=False)
        sigma_p, sigma_m = s[0], s[1]
        return [sigma_p, sigma_m] 

    def _rho_coincidence(self, p1, p2, eta_0, dc_0, R_T, R_J, dc_T, dc_J):
        s_1 = self._get_svd(p1, p2, eta_0, dc_0, R_T, R_J, dc_T, dc_J, [1, 0])
        s_2 = self._get_svd(p1, p2, eta_0, dc_0, R_T, R_J, dc_T, dc_J, [1, 1])
        norm, _, _, _, _ = self.local_proba_model._rho_heralded()
        return norm, s_1, s_2

    def _tr(self, side, cond):
        cond_T, cond_J = cond
        if side == 'A':
            p1, p2 = self.elink.p_A, self.elink.p_B
            eta_elink = self.elink.eta_A
            dc_elink = self.elink.dc_A
        elif side == 'B':
            p1, p2 = self.elink.p_B, self.elink.p_A
            eta_elink = self.elink.eta_B
            dc_elink = self.elink.dc_B
        else:
            raise ValueError("The side must be `A` or `B` for Alice or Bob.")
        # Get parameters
        dc_T, dc_J = (dc_elink + self.dc_T) * cond_T, (dc_elink + self.dc_J) * cond_J
        R_T, R_J = 1 - eta_elink * self.eta_T * cond_T, 1 - eta_elink * self.eta_J * cond_J
        eta_0, dc_0 = self.elink.eta_0, self.elink.dc_0
        # Get the SVDs
        norm, s_1, s_2 = self._rho_coincidence(p1, p2, eta_0, dc_0, R_T, R_J, dc_T, dc_J)
        # Calculation
        factor = (1-dc_T) * (1-dc_J) * (1-p1)*(1-p2) * (1/norm)
        coef_1 = (1-dc_0) / ((1-s_1[0])*(1+s_1[0]) * (1-s_1[1])*(1+s_1[1]))
        coef_2 = (1-dc_0)**2 /((1-s_2[0])*(1+s_2[0]) * (1-s_2[1])*(1+s_2[1]))
        return factor * (coef_1 - coef_2)

    def get_p00(self, side):
        """Get the local Alice:no-click|Bob:no-click probability after heralding.

        Returns:
            float: Probability of no-click at Alice and Bob.
        """
        self.check_integrity()
        return self._tr(side, [1, 1])

    def get_p10(self, side):
        """Get the local Alice:click|Bob:no-click probability after heralding.

        Returns:
            float: Probability of click at Alice and no-click at Bob.
        """
        self.check_integrity()
        return self._tr(side, [0, 1]) - self._tr(side, [1, 1])

    def get_p01(self, side):
        """Get the local Alice:no-click|Bob:click probability after heralding.

        Returns:
            float: Probability of no-click at Alice and click at Bob.
        """
        self.check_integrity()
        return self._tr(side, [1, 0]) - self._tr(side, [1, 1])

    def get_p11(self, side):
        """Get the local Alice:click|Bob:click probability after heralding.

        Returns:
            float: Probability of click at Alice and Bob.
        """
        self.check_integrity()
        return 1 - self._tr(side, [0, 1]) - self._tr(side, [1, 0]) + self._tr(side, [1, 1])

    def get_proba(self, side):
        """Get all the probabilities.

        Returns:
            tuple: Tuple of probabilities (P00, P01, P10, P11, P_herald).
        """
        return self.get_p00(side), self.get_p01(side), self.get_p10(side), self.get_p11(side)

        




class EntWitness():
    # TODO
    def __init__(self):
        self.elink = ElemLink()
        self.local_proba_rho = LocalProbaModel(self.elink)
        self.local_proba_obs = LocalProbaModel(self.elink)
        self.local_proba_coinc_A = LocalProbaCoincidence(self.elink)
        self.local_proba_coinc_B = LocalProbaCoincidence(self.elink)

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
        print(f"z_min = {z_min} \nz_max = {z_max}")
        z_min, z_max = (min(z_min, 0), 0) if mask else (z_min, z_max)
        
        cmap = plt.get_cmap('viridis_r')
        normalizer = colors.Normalize(vmin=z_min, vmax=z_max)
        fig, ax = plt.subplots(1, layout="constrained")
        cs = ax.contourf(X, Y, Z, levels=N_lvl, cmap=cmap, norm=normalizer)
        ax.set_xlabel(r'Alpha', fontsize=12)
        ax.set_ylabel(r'Beta', fontsize=12)
        ax.grid()
        ccf = cm.ScalarMappable(norm=normalizer, cmap=cmap)
        cbar = fig.colorbar(ccf, ax=ax)
        cbar.set_label('$w_{ppt}-<W>$')
        return fig

