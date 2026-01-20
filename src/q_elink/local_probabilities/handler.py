from ._imports import *
from scipy.stats import norm

# Internal libs
from .model import LocalProbabilityModel

class FitResultAnalyzer:
    def __init__(self, data_raw=None):
        self.data_raw = data_raw
        self.data_processed = None
        pass

    
    def _check_processed_integrity(self):
        """
        Check if the data has been loaded and prcessed.
        TODO
        """
        if self.data_processed is None:
            raise ValueError("Processed data not set, please set processed data with `self.load_data_processed() or `self.build_data()` first.")

    def _check_raw_integrity(self):
        """
        Check if raw data has been loaded.
        TODO: check dictionary tree intergity.
        """
        if self.data_raw is None:
            raise ValueError("Raw data not set, please set raw data with `self.load_data_raw()` first.")

    def load_data_raw(self, path):
        """
        Loads raw data JSON file from `ProbabilityFitter` and store it in `self.data_raw`.
        """
        with open(path, 'r') as f:
            data_raw = json.load(f)
        self.data_raw = data_raw
            
    def load_data_processed(self, path):
        """
        Loads processed data JSON file from `FitResultAnalyzer` and store it in `self.data_processed`.
        """
        with open(path, 'r') as f:
            data_processed = json.load(f)
        self.data_processed = data_processed

    def build_data(self):
        # Initialization
        self._check_raw_integrity()
        data_raw = self.data_raw
        name_fit = data_raw['fit_param']['fit_param_name']
        mse_init = []
        mse_fitted = []
        data_init_dict = {name: [] for name in name_fit}
        data_fitted_dict = {name: [] for name in name_fit}
        # Span all the fits in the fit result dictionnary
        for _, fit in data_raw['all_fit_result'].items():
            if 'error' not in fit['fit_result']:  # If no error in the fit
                data_init = fit['init_param']  # Get the set of initial parameter of the fit
                data_fitted = fit['fit_result']['x']  # Get the set of optimized parameter of the fit
                mse_init.append(fit['fit_result']['init_cost'])
                mse_fitted.append(fit['fit_result']['cost'])
                for i, name in enumerate(name_fit):  # Span and store the parameters according to the `name_fit` order
                    data_init_dict[name].append(data_init[i])
                    data_fitted_dict[name].append(data_fitted[i])
        # Draw statistics
        statistics = {}
        for name, fit_values in data_fitted_dict.items():
            statistics[name] = {}
            statistics[name]["mean"] = np.mean(fit_values)
            statistics[name]["std"] = np.std(fit_values)
        # Store data
        self.data_processed = {
            'statistics': statistics,
            'data_exp': data_raw['data'],
            'data_init': data_init_dict,
            'data_fitted': data_fitted_dict,
            'mse_init': mse_init,
            'mse_fitted': mse_fitted,
        }

    def get_data(self):
        if self.data_processed is not None:
            return self.data_processed
        else:
            raise ValueError("Please first build the processed data with `build_data()`.")

    def get_stat(self):
        if self.data_processed is not None:
            return self.data_processed['statistics']
        else:
            raise ValueError("Please first build the processed data with `build_data()`.")

    def save_data(self, save_path):
        """
        Saves a dictionary to a JSON file.
        It handles numpy arrays and scipy.optimize.OptimizeResult objects.

        Args:
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

        data_dict = self.data_processed
        with open(save_path, 'w') as f:
            json.dump(data_dict, f, cls=NpEncoder, indent=4)


    def plot_fit(self, param_gauss={}, n_sample=1000, n_sigma=1, smooth_plot=False, seed=42):

        def wrapper(p_A, p_B, local_proba):
            local_proba.set_pump(p_A, p_B)
            # local_proba.show()
            p00, p01, p10, p11 = local_proba.get_proba()
            return p00, p01, p10, p11
            
        # Set elink model
        elink = ElementaryLink()
        local_proba = LocalProbabilityModel(elink)

        # Get and build pimp parameters
        data_processed = self.data_processed
        data_exp = data_processed["data_exp"]
        pij_exp_dict = data_exp["proba_data"]
        pij_exp = np.array([pij for _, pij in pij_exp_dict.items()])
        pij_exp = np.transpose(pij_exp) # to get pij_exp[:, i]
        p_A, p_B = np.array(data_exp["pump_data"]["p_A"]), np.array(data_exp["pump_data"]["p_B"])
        p_mean = (p_A + p_B)/2
        n_exp = len(p_A)

        # Set fit parameters
        param_gauss = self.data_processed['statistics'] | param_gauss
        param_dict = { key: float(value["mean"]) for key, value in param_gauss.items() }
        print(param_gauss)

        # Get fit points
        elink.set_param(param_dict, log_dc=True)
        vect_wrapper = np.vectorize(lambda p_A, p_B: wrapper(p_A, p_B, local_proba))
        pij_tuple = vect_wrapper(p_A, p_B)
        pij_fit = np.column_stack(pij_tuple) 

        # Monte-Carlo
        rng = np.random.default_rng(seed)
        data_sample = np.zeros((n_sample, n_exp, 4))    # [n_sample, n_exp, n_pij]
        sample = {}
        for i in tqdm(range(n_sample)):
            # Sample the elink parameters
            for param_name, param_stat in param_gauss.items():
                mu, sigma = param_stat["mean"], param_stat["std"]
                sample[param_name] = rng.uniform(mu - n_sigma*sigma, mu + n_sigma*sigma)
            # Get the pij for every pump parameter
            elink.set_param(sample, log_dc=True)
            vect_wrapper = np.vectorize(lambda p_A, p_B: wrapper(p_A, p_B, local_proba))
            pij_tuple = vect_wrapper(p_A, p_B)
            pij = np.column_stack(pij_tuple)
            data_sample[i, :, :] = pij

        # Get the min/max over all the Monte-Carlo samples
        pij_max = np.max(data_sample, axis=0)
        pij_min = np.min(data_sample, axis=0)

        # Build parameters for the plot
        p_all = np.concatenate((p_A, p_B))
        p_min, p_max = np.min(p_all), np.max(p_all)
        margin = (p_max - p_min)*0.1
        x_min = p_min - margin
        x_max = p_max + margin
        x_pump = np.linspace(x_min, x_max, 50)
        subplot_name = [r'$P_{00}$', r'$P_{01}$', r'$P_{10}$', r'$P_{11}$']
        fig, axs = bst.subplots_grid(2, 2, 4, plot_size=(5*0.9, 4*0.9), grid=True)

        # Plot: smooth continious plot via extrapolation
        if smooth_plot:
            # Sort the p_mean for `UnivariateSpline`
            sort_idx = np.argsort(p_mean)
            p_mean_sorted = p_mean[sort_idx]
            pij_fit_sorted = pij_fit[sort_idx]
            pij_min_sorted = pij_min[sort_idx]
            pij_max_sorted = pij_max[sort_idx] 
            x_pump = np.linspace(x_min, x_max, 50)
            # Span the plots
            for i, ax in enumerate(axs):
                # Get the min and max
                pij_fit_col = pij_fit_sorted[:, i]
                pij_min_col = pij_min_sorted[:, i]
                pij_max_col = pij_max_sorted[:, i]
                # Build the smooth function
                k, s = 1, 10
                spl_fit = sp.interpolate.UnivariateSpline(p_mean_sorted, pij_fit_col, k=k, s=s)
                spl_min = sp.interpolate.UnivariateSpline(p_mean_sorted, pij_min_col, k=k, s=s)
                spl_max = sp.interpolate.UnivariateSpline(p_mean_sorted, pij_max_col, k=k, s=s)
                y_fit_smooth = spl_fit(x_pump)
                y_min_smooth = spl_min(x_pump)
                y_max_smooth = spl_max(x_pump)
                # Plot
                ax.fill_between(x_pump, y_min_smooth, y_max_smooth, color='gray', alpha=0.3, label=fr"${n_sigma} \cdot \sigma$")
                ax.plot(x_pump, y_fit_smooth, 'k--', label="Fit")
                ax.plot(p_mean, pij_exp[:, i], '.', label="Data")
                ax.set_xlabel("Pump parameter")
                ax.set_ylabel(subplot_name[i])

            handles, labels = axs[0].get_legend_handles_labels()
            fig.legend(handles, labels, loc="lower center", ncol=len(labels), bbox_to_anchor=(0.5, -0.05))
            # plt.tight_layout(rect=[0, 0.05, 1, 1])
            return fig, axs

        # Plot: disctrete shots plot
        else:
            for i, ax in enumerate(axs):
                y = pij_fit[:, i]
                y_min, y_max = pij_min[:, i], pij_max[:, i]
                yerr = [y - y_min, y_max - y]
                ax.errorbar(p_mean, y, yerr=yerr, fmt='+', capsize=3)
                # ax.plot(p_mean, pij_fit[;, i], 'o')
                # ax.plot(p_mean, pij_min[:, i], 'o')
                # ax.plot(p_mean, pij_max[:, i], '.')
                ax.plot(p_mean, pij_exp[:, i], 'o')
            return fig, axs



    def plot_shots(self):
        self._check_processed_integrity()
        # Get processed data
        data_processed = self.data_processed
        data_init = data_processed["data_init"]
        data_fitted = data_processed["data_fitted"]
        mse_init = data_processed["mse_init"]
        mse_fitted = data_processed["mse_fitted"]
        # Initialiaze the plit
        n_plot = len(data_init) + 1
        fig, axs = bst.subplots_grid(3, 3, n_plot, plot_size=(5, 4))

        # Plot the e-link parameters
        for i, (name, _) in enumerate(data_init.items()):
            ax = axs[i]
            if name in ['eta_0', 'eta_A', 'eta_B', 'dc_0', 'dc_A', 'dc_B']:
                ax.plot(data_init[name], 'k+', label="initial")
                ax.plot(data_fitted[name], '+', color='darkorange', label="fit")
            else:
                raise ValueError(f"Parameter {name} not known.")
            ax.set_ylabel(name, fontsize=14)
            ax.set_xlabel("$N$", fontsize=14)
            ax.set_title(f"Fit of {name}")
            ax.legend(loc=1) 
            ax.grid()
        # Plot MSE
        axs[-1].plot(mse_init, 'k+', label="initial")
        axs[-1].plot(mse_fitted, '+', color='darkorange', label="fit")
        axs[-1].set_title("MSE")
        axs[-1].set_xlabel("$N$", fontsize=14)
        axs[-1].set_ylabel("MSE", fontsize=14)
        axs[-1].legend(loc=1)
        axs[-1].grid()
        return fig, axs
        
    # Build and plot histogram function
    def plot_histogram(self, param_plot={}, param_gauss={}):
        self._check_processed_integrity()
        # Get the processed data
        data_processed = self.data_processed
        data_fitted = data_processed['data_fitted']
        # Set param gauss
        param_gauss = self.data_processed['statistics'] | param_gauss

        # Build default param_plot
        for name, stats in param_gauss.items():
            mu, sigma = stats["mean"], stats["std"]
            default_param = {
                "bounds": [mu - 3*sigma, mu + 3*sigma],
                "bins": 20,
                "plot_name": name,
                "digits": 2,
            }
            if name in param_plot:
                param_plot[name] = default_param | param_plot[name]
            else:
                param_plot[name] = default_param

        # Initialize the plot
        n_plot = len(param_gauss)
        # fig, axs = plt.subplots(n_plot, 1, figsize=(6, 4 * n_plot), squeeze=False)
        # axs = axs.ravel()
        fig, axs = bst.subplots_grid(2, 3, n_plot, plot_size=(5, 4))
        # Plot the parameters
        for i, (name, _) in enumerate(param_gauss.items()):
            # Get single plot param and data
            ax = axs[i]
            plot_data = data_fitted[name]
            normal_dist = param_gauss[name]
            plot_info = param_plot[name]
            mu, sigma = normal_dist["mean"], normal_dist["std"]
            x_min, x_max = plot_info["bounds"]
            bins = plot_info["bins"]
            plot_name = plot_info["plot_name"]

            # Plot histogram
            counts, _, patches = ax.hist(plot_data, bins=bins, density=True)
            counts /= np.max(counts)
            for count, patch in zip(counts, patches):
                patch.set_height(count)

            # Normal distribution
            x = np.linspace(x_min, x_max, 100)
            y_norm = norm.pdf(x, mu, sigma)
            if np.max(y_norm) > 0:
                y_norm /= np.max(y_norm)
            label = bst.sci_notation(mu, sigma, n=plot_info["digits"])
            ax.plot(x, y_norm, 'k--', linewidth=2, label='Normal dist. \n' + label)

            ax.set_ylim([0, 1.2])
            ax.set_xlim([x_min, x_max])
            ax.set_xlabel(plot_name, fontsize=14)
            ax.set_title(f"Fit of {plot_name}")
            ax.legend(loc=1)
            ax.grid()
        return fig, axs

