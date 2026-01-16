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
        """
            Set elementary link parameters from input dictionary.
        """
        allowed_keys_ass = [ "p_A", "p_B", "eta_0", "eta_A", "eta_B", "dc_0", "dc_A", "dc_B" ]
        allowed_keys_sym = [ "p", "eta", "dc" ]
        log_keys = [ "dc", "dc_A", "dc_B", "dc_0" ]
        for key, value in param_dict.items():
            # Assymetric parameters
            if key in allowed_keys_ass:
                if log_dc and key in log_keys:
                    value = np.power(10, value)
                setattr(self, key, value)
            # Symmetric parameters
            elif key in allowed_keys_sym:
                if log_dc and key in log_keys:
                    value = np.power(10, value)
                setattr(self, key+"_A", value)
                setattr(self, key+"_B", value)
            # Key not known
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
