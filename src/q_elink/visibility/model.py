from ._imports import *

class InterferenceModel():
    def __init__(self, the_elemLink: ElementaryLink) -> None:
        self.elink = the_elemLink
        self.local_proba = LocalProbabilityModel(the_elemLink)
        self.eta_A = 1
        self.eta_B = 1
        self.dc_A = 0
        self.dc_B = 0


    def set_param(self, param_dict):
        """Set parameters from dictionary."""
        allowed_keys = ["eta_A", "eta_B", "dc_A", "dc_B"]
        for key in param_dict:
            if key in allowed_keys:
                setattr(self, key, param_dict[key])
            else:
                warnings.warn(f"Uknown parameter: {key}.", UserWarning)
    

    def check_integrity(self):
        #self.elink.check_integrity()
        pass

    def _get_svd(self, phi, cond):
        """
        Compute SVD for one term of the probability sum.

        Args:

        Returns:

        """
        # Get the pump parameter
        p_A, p_B = self.elink.p_A, self.elink.p_B
        # Get the efficiencies for the modes a,b,c and d
        eta_b, eta_c = self.elink.eta_0, self.elink.eta_0
        eta_a, eta_d = self.elink.eta_A * self.eta_A, self.elink.eta_B * self.eta_B
        # Build the loss values for the modes a, b, c and d
        cond_a, cond_b, cond_c, cond_d = cond
        R_b, R_c = 1 - cond_b * eta_b, 1 - cond_c * eta_c
        R_a, R_d = 1 - cond_a * eta_a, 1 - cond_d * eta_d
        # Matrix of the modes a and d
        M_ad = np.array([
            [np.sqrt(p_A*R_a) * np.exp(1j * phi),  np.sqrt(p_A*R_d) * np.exp(1j * phi)],
            [np.sqrt(p_B*R_a), - np.sqrt(p_B*R_d)]
            ]) / np.sqrt(2)
        # Matrix of the modes b and c
        M_bc = np.array([
            [np.sqrt(R_b),   np.sqrt(R_c)],
            [np.sqrt(R_b), - np.sqrt(R_c)]
            ]) / np.sqrt(2)
        # Matrix
        M = np.matmul(np.transpose(M_ad), M_bc)
        # SVD calculation
        svd = np.linalg.svd(M, full_matrices=False, compute_uv=False)
        return svd

    def _get_darkcount(self, cond):
        cond_a, cond_b, cond_c, cond_d = cond
        dc_b, dc_c = self.elink.dc_0, self.elink.dc_0
        dc_a, dc_d = self.elink.dc_A + self.dc_A, self.elink.dc_B + self.dc_B
        kappa_b, kappa_c = 1 - cond_b * dc_b, 1 - cond_c * dc_c
        kappa_a, kappa_d = 1 - cond_a * dc_a, 1 - cond_d * dc_d
        return kappa_a * kappa_b * kappa_c * kappa_d

    def _get_norm(self):
        N = self.local_proba.get_p_herald()
        return N

    def _get_coincidence_term(self, phi, cond):
        sigma_p, sigma_m = self._get_svd(phi, cond)
        return 1/(1-np.abs(sigma_p)**2) * 1/(1-np.abs(sigma_m)**2)
        
    def get_proba_coincidence(self, phi):
        terms_dict = {
                "term1": { # + Ra Rb
                    "cond": [1, 1, 0, 0],
                    "sign": 1
                    },
                "term2": { # - Ra Rb Rc
                    "cond": [1, 1, 1, 0],
                    "sign": -1
                    },
                "term3": { # - Ra Rb Rd
                    "cond": [1, 1, 0, 1],
                    "sign": -1
                    },
                "term4": { # + Ra Rb Rc Rd
                    "cond": [1, 1, 1, 1],
                    "sign": 1
                    },
                }
        proba_coincidence = 0
        for _, term in terms_dict.items():
            cond = term["cond"]
            sign = term["sign"]
            darkcount = self._get_darkcount(cond)
            coincidence_term = self._get_coincidence_term(phi, cond)
            proba_coincidence += sign * darkcount * coincidence_term
        norm = self._get_norm()
        p_A, p_B = self.elink.p_A, self.elink.p_B
        spdc_factor = (1 - p_A) * (1 - p_B)
        return spdc_factor/norm * proba_coincidence


    def get_visibility(self, N=50, eps=1e-1):
        pi_bound = np.pi + eps
        phi_list = np.linspace(-pi_bound, pi_bound, N)
        proba_coincidence_values = [self.get_proba_coincidence(phi) for phi in phi_list]
        proba_min, proba_max = np.min(proba_coincidence_values), np.max(proba_coincidence_values)
        visibility = proba_max - proba_min
        return visibility

    def get_proba_coincidence_pert(self, phi):
        R_A = 1 - self.elink.eta_A
        kappa_A, kappa_B = 1 - self.elink.dc_A, 1 - self.elink.dc_B
        factor = kappa_B * (1 - kappa_A * R_A)
        return factor * np.cos(phi/2)**2

    def get_visibility_pert(self, phi):
        eta = self.elink.eta_0
        return eta/4


    def plot_proba_coincidence(self, phi_bound=(-np.pi, np.pi), N=50, perturbative=True, param_dict=None):
        X = np.linspace(*phi_bound, N)
        if param_dict is None:
            fig, axs = bst.subplot_grid(1, 1, 1)
            Y = [ self.get_proba_coincidence(x) for x in X ]
            axs[0].plt(X, Y)
        # else:
        #
        #     def foo(x, param):
        #         self.elink.set_param({ param_name: param})
        #         proba_coincidence  = self.get_proba_coincidence(x)
        #         return proba_coincidence
        #
        #     fig, axs = bst.su

        return fig, axs

