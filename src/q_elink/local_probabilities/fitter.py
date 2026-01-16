from ._imports import *

from scipy.optimize import least_squares
from scipy.stats import norm




class ProbabilityFitter():
    def __init__(self, rng_seed = None):
        self.rng_seed = rng_seed
        self.rng = np.random.default_rng(self.rng_seed)

        # Data (set by the user via `set_model` and `set_data` and `set_init_value`)
        self.local_proba_mod = None
        self.local_proba_exp = None
        self.init_val_dict = None

        # Fit parameter (set by the user via `set_fit`)
        self.n_fit = None
        self.pij_pond = [1 for _ in range(4)]
        self.relative = False
        self.ftol = 1e-08
        self.xtol = 1e-08
        self.gtol = 1e-08
        
        # Set the model
        self.elink = ElementaryLink()
        self.local_proba_mod = LocalProbabilityModel(self.elink)

    # Check intergity methods

    def check_integrity(self):
        self._check_fit_integrity()
        self._check_init_val_integrity()
        self._check_model_integrity()
        self._check_exp_integrity()

    def _check_model_integrity(self):
        """ ToDo """
        pass

    def _check_exp_integrity(self):
        """ ToDo """
        pass


    def _check_init_val_integrity(self):
        """ Check init_val_dict integrity. """
        if self.init_val_dict is None:
            raise ValueError("Please set initial values with `set_init_value`.")
        for key, value in self.init_val_dict.items():
            if not isinstance(value, (int, float, list, np.ndarray)):
                raise ValueError("Boundaries from initial condition must be list, numpy array, int or float.")

    def _check_fit_integrity(self):
        """ Check fit integrity before fit """
        if self.local_proba_exp is None:
            raise ValueError("Please set the experimental data first with `set_data`.")
        if not(isinstance(self.n_fit, int)):
            raise ValueError("Please set the number of fit via `set_fit` first.")

    # Build methods

    # def _build_rng(self):
    #     old_rng = self.rng
    #     new_rng = old_rng.spawn(1)
    #     self.rng = new_rng[0]

    def _build_param_name(self):
        self._fit_param_name = []
        self._fixed_param_name = []
        for key, value in self.init_val_dict.items():
            if isinstance(value, (list, np.ndarray)):
                self._fit_param_name.append(key)
            elif isinstance(value, (int, float)):
                self._fixed_param_name.append(key)
            else:
                raise ValueError("Boundaries from initial condition must be list, numpy array, int or float.")

    def _build_exp_mat(self):
        """
        Aim to build the experimental data (4 x _n_data) matrix:
            - 4: for the p_ij
            - _n_data: for the number of points
        """
        data_exp_mat = np.zeros((self._n_data, 4))
        data_exp_dict = self.local_proba_exp.proba_dict
        for i, key in enumerate(["p00", "p01", "p10", "p11"]):
            data_exp_mat[:,i] = data_exp_dict[key]
        self._data_exp_mat = data_exp_mat.T


    # Set parameters methods

    def _set_fixed_param(self):
        """ 
        Set fixed parameters in the elink instance.
        The param input must follow the same order than _fixed_param_name list.
        """
        for name in self._fixed_param_name:
            setattr(self.elink, name, self.init_val_dict[name])
        
    def _set_model_param(self, param):
        """ 
            Set fit parameters in the elink instance.
            The param input must follow the same order than _fit_param_name list.
            """
        # for name, value in zip(self._fit_param_name, param):
        #     if name in ['dc_0', 'dc_A', 'dc_B']:
        #         value = np.power(10, value)
        #     setattr(self.elink, name, value) # ToDo: use the elink.set_param() method instead
        self.elink.set_param(param, log_dc=True)


    # Initial value sampling method

    def _get_init_val(self):
        param_init, param_min, param_max = [], [], []
        # NB: respawn rng not needed when sequentiel -> rng i updated after each call
        # self._build_rng()
        # rng = self.rng
        for param_name in self._fit_param_name:
            value = self.init_val_dict[param_name]
            param_min.append(value[0])
            param_max.append(value[1])
            if param_name in ['eta_0', 'eta_A', 'eta_B', 'dc_0', 'dc_A', 'dc_B']:
                param_init.append(rng.uniform(low=value[0], high=value[1]))
            else:
                raise ValueError(f"Fitting parameter {param_name} not available account.")
        return param_init, param_min, param_max 


    # Fitting and residual function (worker function)

    def _get_mod_mat(self, param):
        """
        Get the model matrix for residual calculation
        """
        self._set_model_param(param)
        def wrapper(p_A, p_B):
            self.local_proba_mod.set_pump(p_A=p_A, p_B=p_B)
            return self.local_proba_mod.get_proba()
        data_model_mat = np.array([wrapper(p_A, p_B) for p_A, p_B in zip(self._p_A, self._p_B)])
        return data_model_mat.T        
   

    def _residual(self, param):
        """
        Create two (4 x _n_data) matrices for the experimental data and the model data.
            - 4: for the p_ij
            - _n_data: for the number of points
        Get the difference and collapse according the p_ij axis with mean square method.
        """
        data_mod_mat = self._get_mod_mat(param)
        if self.relative:
            denom = self._data_exp_mat.copy()
            denom[denom == 0] = np.inf
            diff_mat = (data_mod_mat - self._data_exp_mat) / denom
        else:
            diff_mat = data_mod_mat - self._data_exp_mat 
        return np.sqrt(np.sum(self.pij_pond[:, np.newaxis] * (diff_mat**2), axis=0))
        
    
    def _single_fit(self, x_init, x_min, x_max):
        optimized_result = least_squares(self._residual, x_init, bounds=(x_min, x_max), ftol=self.ftol, xtol=self.xtol, gtol=self.gtol)
        return optimized_result

    def get_residual(self):
        self._build_exp_mat()
        buffer = self._fit_param_name if hasattr(self, '_fit_param_name') else []
        self._fit_param_name = []
        try:
            residus = self._residual([])
        finally:
            self._fit_param_name = buffer
        return np.sqrt(np.sum(residus**2))

    # Fitting method

    # def set_model(self, local_proba_mod: LocalProbabilityModel):
    #     self._check_model(local_proba_mod)
    #     self.local_proba_mod = local_proba_mod
    #     self.elink = local_proba_mod.elink

    def set_data(self, local_proba_exp: LocalProbaExperiment):
        self._check_exp_integrity()
        self.local_proba_exp = local_proba_exp
        self._p_A = local_proba_exp.pump_dict["p_A"]
        self._p_B = local_proba_exp.pump_dict["p_B"]
        self._n_data = len(self._p_A)

    def set_init_value(self, init_val_dict):
        self.init_val_dict = init_val_dict

    def set_fit(self, n_fit, relative=False, pij_pond=[1 for el in range(4)], ftol=1e-08, xtol=1e-08, gtol=1e-08):
        self.n_fit = n_fit
        self.relative = relative
        self.pij_pond = np.array(pij_pond)
        self.ftol = ftol
        self.xtol = xtol
        self.gtol = gtol


    def fit(self):

        def worker():
            param_init, param_min, param_max = self._get_init_val()
            try:
                fit_result = self._single_fit(param_init, param_min, param_max)
                init_cost = np.sqrt(np.sum(self._residual(param_init)))
                # Construction manuelle 
                result_data = {
                    'success': fit_result.success,
                    'status': fit_result.status,
                    'message': fit_result.message,
                    'x': fit_result.x,
                    'cost': fit_result.cost,
                    "init_cost": init_cost,
                    'fun': fit_result.fun, # Les résidus
                    'nfev': fit_result.nfev, # Nombre d'évaluations de la fonction
                }
            except Exception as e:
                return {
                        'init_param': param_init,
                        'fit_result': { "error": str(e) } ,
                }
            return {
                    'init_param': param_init,
                    'fit_result': result_data,
            }

        print("Initialization...")
        self._check_fit_integrity()
        self._check_init_val_integrity()
        self._build_param_name()
        self._build_exp_mat()
        self._set_fixed_param()
        print("Begining of the fit...") 
        result_dict = {
            'data': {
                'pump_data': self.local_proba_exp.pump_dict,
                'proba_data': self.local_proba_exp.proba_dict,
            },
            'fit_param': {
                'fit_param_name': self._fit_param_name,
                'fixed_param_name': self._fixed_param_name,
                'param_bound': self.init_val_dict,
                'n_fit': self.n_fit,
                'pij_pond': self.pij_pond,
                'relative': self.relative,
                'ftol': self.ftol,
                'xtol': self.xtol,
                'gtol': self.gtol,
            },
            'all_fit_result': { },
        }
        for i in tqdm(range(self.n_fit)):
            result_dict["all_fit_result"][f"fit_{i}"] = worker()
        print("Fit ended.")
        self.result_dict = result_dict

        return result_dict

    def save_result(self, savepath):
        """
        Saves a dictionary to a JSON file.
        It handles numpy arrays and scipy.optimize.OptimizeResult objects.

        Args:
            data_dict (dict): The dictionary to save.
            savepath (str): The path to save the JSON file.
        """
            
        # JSON personal encoder for numpy variables
        class NpEncoder(json.JSONEncoder):
            def default(self, obj):
                if isinstance(obj, np.integer):
                    return int(obj)
                if isinstance(obj, np.floating):
                    return float(obj)
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                return super(NpEncoder, self).default(obj)

        result_dict = self.result_dict

        with open(savepath, 'w') as f:
            json.dump(result_dict, f, cls=NpEncoder, indent=4)





