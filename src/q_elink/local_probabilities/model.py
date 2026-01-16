from ._imports import *

class LocalProbabilityModel():
    def __init__(self, the_elemLink: ElementaryLink) -> None:
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




class CoincidenceProbabilityModel():
    def __init__(self, elink: ElementaryLink):
        self.eta_T = None
        self.eta_J = None
        self.dc_T = None
        self.dc_J = None
        self.side = None
        self.elink = elink
        self.local_proba_model = LocalProbabilityModel(self.elink)


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
