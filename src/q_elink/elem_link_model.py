from ._imports import *

@dataclass
class ElemLink():
    # pump parameters
    p_A: float = None
    p_B: float = None
    # efficiencies
    eta_0: float = None
    eta_A: float = None
    eta_B: float = None
    # dark-count
    dc_0: float = None
    dc_A: float = None
    dc_B: float = None

    def set_param(self, param_dict, log_dc=False):
        """Set parameters from dictionary."""
        allowed_keys = [ "p_A", "p_B", "eta_0", "eta_A", "eta_B", "dc_0", "dc_A", "dc_B" ]
        for key in param_dict:
            if key in allowed_keys:
                if log_dc and key in [ "dc_0", "dc_A", "dc_B" ]: 
                    setattr(self, key, np.power(10, param_dict[key]))
                else:
                    setattr(self, key, param_dict[key])
            else:
                warnings.warn(f"Uknown parameter: {key}.", UserWarning)

    def set_pump(self, p_A=None, p_B=None):
        bst.set_not_none(self, p_A=p_A, p_B=p_B)


    def check_integrity(self):
        """Check if the parameters have been set before any calculation.

        Raises:
            ValueError: If any required parameter is not set.
        """
        eff_param = {
            'eta_0': self.eta_0,
            'eta_A': self.eta_A,
            'eta_B': self.eta_B,
            'dc_0': self.dc_0,
            'dc_A': self.dc_A,
            'dc_B': self.dc_B,
        }
        pump_param = {
            'p_A': self.p_A,
            'p_B': self.p_B,
        }
        for name, value in pump_param.items():
            if value is None:
                raise ValueError(f"Parameter {name} has not been set, please set {name} first.")
        for name, value in eff_param.items():
            if value is None:
                raise ValueError(f"Parameter {name} has not been set, please set {name} first.")

class LocalProbaModel():
    def __init__(self, the_elemLink: ElemLink) -> None:
        self.elink = the_elemLink
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
    
    def set_pump(self, p_A=None, p_B=None):
        self._set_not_none(self.elink, p_A=p_A, p_B=p_B)

    def check_integrity(self):
        #self.elink.check_integrity()
        pass

    def _set_not_none(self, class_parent, **kwargs):
        for name, val in kwargs.items():
            if val is not None:
                setattr(class_parent, name, val) 

    def _get_svd(self, p_A, p_B, eta_0, cond):
        """Compute the SVDs for probabilities computation.

        Args:
            cond_1 (int 0,1): Conditonal efficiency 1
            cond_2 (int 0,1): Conditional efficiency 2

        Returns:
            tuple: List SVD values, list unitary transformations coefficient

        """
        cond_1, cond_2 = cond
        R_1, R_2 = 1 - cond_1 * eta_0, 1 - cond_2 * eta_0
        M = np.array([[np.sqrt(p_A * R_2),  np.sqrt(p_A * R_1)],
                      [-np.sqrt(p_B * R_2), np.sqrt(p_B * R_1)]]) / np.sqrt(2)
        A, B, C = np.linalg.svd(M, full_matrices=True, compute_uv=True, hermitian=False)
        sigma_p, sigma_m = B[0], B[1]
        delta, tau = A[0, 0], A[0, 1]
        return [sigma_p, sigma_m], [delta, tau]

    def _normalization(self, p_A, p_B, dc_0, S_1, S_2):
        """Get the normalization constant for the heralding state.

        Args:
            S_1 (list): List of two elements corresponding to the SVDs sigma/nu_plus/minus.
            S_2 (list): List of two elements corresponding to the SVDs sigma/nu_plus/minus.

        Returns:
            float: Normalization constant of the heralded state

        """
        a = (1 - p_A) * (1 - p_B)
        b = (1 - dc_0) * (1-S_2[0])*(1+S_2[0]) * (1-S_2[1])*(1+S_2[1]) - ((1 - dc_0)**2) * (1-S_1[0])*(1+S_1[0]) * (1-S_1[1])*(1+S_1[1])
        c = (1-S_1[0])*(1+S_1[0]) * (1-S_1[1])*(1+S_1[1]) * (1-S_2[0])*(1+S_2[0]) * (1-S_2[1])*(1+S_2[1])
        return a * b / c

    def _rho_heralded(self):
        """Get the heralded state's SVD and normalization.

        Args:
            /

        Returns:
            tuple: normalization constant, SVD values, unitary transformations
        """
        p_A, p_B = self.elink.p_A, self.elink.p_B
        eta_0 = self.elink.eta_0
        dc_0 = self.elink.dc_0
        S_1, U_1 = self._get_svd(p_A, p_B, eta_0, [1, 0])
        S_2, U_2 = self._get_svd(p_A, p_B, eta_0, [1, 1])
        N = self._normalization(p_A, p_B, dc_0, S_1, S_2)
        return N, S_1, S_2, U_1, U_2

    def _f(self, n_1, n_2, delta, tau, eta_A, eta_B, dc_A, dc_B, alpha, beta):
        """Function used for trace computation.

        Args:
            n_1 (float): Mean number of photons in the 1st thermal state.
            n_2 (float): Mean number of photons in the 2nd thermal state.
            tau (float): Unitary transformation coefficient from SVD
            delta (float): Unitary transformation coefficient from SVD.
            eta_A (float): Efficiency at Alice's side.
            eta_B (float): Efficiency at Bob's side.
            dc_A (float): Alice's dark-count.
            dc_B (float): Bob's dark-count.
            alpha (float): Displacement operator Alice's side.
            beta (float): Displacement operator Bob's side.

        Returns:
            float: Computed value for trace computation.
        """
        den = (1+n_1*(delta**2*eta_A+tau**2*eta_B))*(1+n_2*(tau**2*eta_A+delta**2*eta_B))-tau**2*delta**2*n_1*n_2*(eta_A-eta_B)**2
        a = np.exp(-(1/den)*(eta_A*np.abs(alpha)**2+eta_B*np.abs(beta)**2))
        b = np.exp(-(n_1/den)*(eta_A*eta_B)*np.abs(tau*alpha-delta*beta)**2)
        c = np.exp(-(n_2/den)*(eta_A*eta_B)*np.abs(delta*alpha+tau*beta)**2)
        return (1-dc_A)*(1-dc_B)*(1/den)*a*b*c


    def _tr(self, alpha, beta, cond):
        """Trace computation to get the probability.

        Args:
            cond_A (bit): Alice POVM parameter
            cond_B (bit): Bob POVM parameter
        Returns:
            float: Computed probability.
        """

        # Pump parameters
        p_A, p_B = self.elink.p_A, self.elink.p_B 
        eta_0, dc_0 = self.elink.eta_0, self.elink.dc_0
        # Alice/bob parameters
        eta_A, eta_B = self.elink.eta_A * self.eta_A, self.elink.eta_B * self.eta_B
        dc_A, dc_B = self.elink.dc_A + self.dc_A, self.elink.dc_B + self.dc_B
        # Conditionnal efficiencies from POVM parameter
        cond_A, cond_B = cond
        eta_A, dc_A = eta_A * cond_A, dc_A * cond_A
        eta_B, dc_B = eta_B * cond_B, dc_B * cond_B
        alpha, beta = alpha * cond_A, beta * cond_B
        # Get the main information about rho_cond
        N, S_1, S_2, U_1, U_2 = self._rho_heralded()
        # Compute the trace
        n = lambda x: x**2 / ((1-x)*(1+x))
        factor_main = (1 - p_A) * (1 - p_B)
        factor_1 = (1 - dc_0) / ((1-S_1[0])*(1+S_1[0]) * (1-S_1[1])*(1+S_1[1]))
        factor_2 = (1 - dc_0)**2 / ((1-S_2[0])*(1+S_2[0]) * (1-S_2[1])*(1+S_2[1]))
        answer = factor_main * (factor_1 * self._f(n(S_1[0]), n(S_1[1]), U_1[0], U_1[1], eta_A, eta_B, dc_A, dc_B, alpha, beta) -
                                factor_2 * self._f(n(S_2[0]), n(S_2[1]), U_2[0], U_2[1], eta_A, eta_B, dc_A, dc_B, alpha, beta))
        return answer / N


    def get_p00(self, alpha=0, beta=0):
        """Get the local Alice:no-click|Bob:no-click probability after heralding.

        Returns:
            float: Probability of no-click at Alice and Bob.
        """
        self.check_integrity()
        return self._tr(alpha, beta, [1, 1])

    def get_p01(self, alpha=0, beta=0):
        """Get the local Alice:click|Bob:no-click probability after heralding.

        Returns:
            float: Probability of click at Alice and no-click at Bob.
        """
        self.check_integrity()
        return self._tr(alpha, beta, [1, 0]) - self._tr(alpha, beta, [1, 1])

    def get_p10(self, alpha=0, beta=0):
        """Get the local Alice:no-click|Bob:click probability after heralding.

        Returns:
            float: Probability of no-click at Alice and click at Bob.
        """
        self.check_integrity()
        return self._tr(alpha, beta, [0, 1]) - self._tr(alpha, beta, [1, 1])

    def get_p11(self, alpha=0, beta=0):
        """Get the local Alice:click|Bob:click probability after heralding.

        Returns:
            float: Probability of click at Alice and Bob.
        """
        self.check_integrity()
        return 1 - self._tr(alpha, beta, [0, 1]) - self._tr(alpha, beta, [1, 0]) + self._tr(alpha, beta, [1, 1])

    def get_p_herald(self, alpha=0, beta=0):
        """Get the heralding click/no-click probability.

        Returns:
            float: Heralding probability.
        """
        self.check_integrity()
        N, _, _, _, _ = self._rho_heralded()
        return N

    def get_proba(self, alpha=0, beta=0):
        """Get all the probabilities.

        Returns:
            tuple: Tuple of probabilities (P00, P01, P10, P11, P_herald).
        """
        return self.get_p00(alpha, beta), self.get_p01(alpha, beta), self.get_p10(alpha, beta), self.get_p11(alpha, beta)

    def plot_proba(self, p_bound=(1e-6, 1-1e-6), n=100):
        pij_name = [r'$P_{00}$', r'$P_{01}$', r'$P_{10}$', r'$P_{11}$']
        p_min, p_max = p_bound
        p = np.linspace(p_min, p_max, n)
        fig, axs = bst.subplot_grid(2, 2, 4)
        
        proba_list = []
        for val in p:
            self.set_pump(p_A=val, p_B=val)
            proba_list.append(self.get_proba())
        proba_all = np.array(proba_list).T

        for i, ax in enumerate(axs):
            proba = proba_all[i]
            name = pij_name[i]
            ax.plot(p, proba)
            ax.set_xlabel(r'Pump $p$')
            ax.set_ylabel(f'Proba {name}')
            ax.grid()
        return fig, axs



class Visibility():
    def __init__(self, the_elemLink: ElemLink) -> None:
        self.elink = the_elemLink
        self.local_proba = LocalProbaModel(the_elemLink)
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
        R_b, R_c = 1 - cond_c * eta_c, 1 - cond_b * eta_b
        R_a, R_d = 1 - cond_a * eta_a, 1 - cond_d * eta_d
        # Matrix of the modes a and d
        print(phi)
        M_ad = np.array([
            [np.sqrt(p_A) * np.exp(1j * phi) * R_a,  np.sqrt(p_A) * np.exp(1j * phi) * R_d],
            [np.sqrt(p_B) * R_a, - np.sqrt(p_B) * R_d]
            ]) / np.sqrt(2)
        # Matrix of the modes b and c
        M_bc = np.array([
            [R_b, R_b],
            [R_c, - R_c]
            ]) / np.sqrt(2)
        # Matrix
        M = np.matmul(np.transpose(M_ad), M_bc)
        # SVD calculation
        _, svd, _ = np.linalg.svd(M, full_matrices=True, compute_uv=True, hermitian=False)
        return svd

    def _get_darkcount(self, cond):
        cond_a, cond_b, cond_c, cond_d = cond
        dc_b, dc_c = self.elink.dc_0, self.elink.dc_0
        dc_a, dc_d = self.elink.dc_A * self.dc_A, self.elink.dc_B * self.dc_B
        kappa_b, kappa_c = 1 - cond_b * dc_b, 1 - cond_c * dc_c
        kappa_a, kappa_d = 1 - cond_a * dc_a, 1 - cond_d * dc_d
        return kappa_a * kappa_b * kappa_c * kappa_d

    def _get_norm(self):
        N = self.local_proba.get_p_herald()
        return N

    def _get_visibility_term(self, phi, cond):
        sigma_p, sigma_m = self._get_svd(phi, cond)
        return 1/(1-sigma_p) * 1/(1-sigma_m)
        
    def get_visibility(self, phi=0):
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
        visibility = 0
        for _, term in terms_dict.items():
            cond = term["cond"]
            sign = term["sign"]
            darkcount = self._get_darkcount(cond)
            visibility_term = self._get_visibility_term(phi, cond)
            visibility += sign * darkcount * visibility_term
        N = self._get_norm()
        return visibility/N

