from ._imports import *

class FidelityModel():
    def __init__(self, the_elemLink: ElementaryLink) -> None:
        self.elink = the_elemLink

    def check_integrity(self):
        self.elink.check_integrity()

    def _svd_overlaps(self, M):
        """Singular values and Bell overlaps (Sigma_A - Sigma_B)^2/2 of a 2x2 mode matrix
        (package sign convention: mode b carries the opposite sign, cf. _get_svd)."""
        U, s, _ = np.linalg.svd(M)
        overlaps = (U[0, :] - U[1, :]) ** 2 / 2
        return s, overlaps

    def get_fidelity(self):
        """Exact fidelity of the heralded link state rho_link (thesis eq. qia:exa:lnk:fid)."""
        self.check_integrity()
        p_A, p_B = self.elink.p_A, self.elink.p_B
        eta_0, dc_0 = self.elink.eta_0, self.elink.dc_0
        R_0 = 1 - eta_0  # central-station loss (package `R` convention)
        # Collective-mode matrices (package sign convention, cf. _get_svd)
        M_sigma = np.array([[np.sqrt(p_A),        np.sqrt(p_A * R_0)],
                            [-np.sqrt(p_B),       np.sqrt(p_B * R_0)]]) / np.sqrt(2)
        M_omega = np.array([[np.sqrt(p_A * R_0),  np.sqrt(p_A * R_0)],
                            [-np.sqrt(p_B * R_0), np.sqrt(p_B * R_0)]]) / np.sqrt(2)
        sigma, S = self._svd_overlaps(M_sigma)
        omega, W = self._svd_overlaps(M_omega)
        # Thermal envelopes
        C_sigma = (1 - sigma[0]**2) * (1 - sigma[1]**2)
        C_omega = (1 - omega[0]**2) * (1 - omega[1]**2)
        # Fidelity
        prefactor = C_sigma * C_omega / (C_omega - (1 - dc_0) * C_sigma)
        bracket = (S[0]*sigma[0]**2 + S[1]*sigma[1]**2
                   - (1 - dc_0) * (W[0]*omega[0]**2 + W[1]*omega[1]**2))
        return prefactor * bracket
