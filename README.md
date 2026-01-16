# q-elink

A Python library for the modelization of quantum elementary links. It provides tools to simulate, analyze, and fit experimental data related to quantum entanglement and quantum communication.

## Features

- **Quantum Elementary Link Modeling**: Core classes to define and parameterize a quantum elementary link.
- **Local Probability Calculation**: Compute heralded and local detection probabilities.
- **Entanglement Witnessing**: Tools for entanglement witnessing analysis.
- **Data Fitting**: Functionality to fit the theoretical model to experimental data.
- **Visualization**: Built-in plotting functions to visualize results.

## Installation

This project uses [Poetry](https://python-poetry.org/) for dependency management.

1.  Clone the repository:
    ```bash
    git clone https://github.com/your-username/q-elink.git
    cd q-elink
    ```

2.  Install dependencies using Poetry:
    ```bash
    poetry install
    ```

## Basic Usage

Here is a minimal example of how to use the library to define an elementary link, set its parameters, and calculate local probabilities.

```python
import q_elink as ql

# 1. Define the parameters for the elementary link
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

# 2. Create an ElementaryLink instance and set its parameters
elink = ql.ElementaryLink()
elink.set_param(param_elink)

# 3. Create a LocalProbabilityModel for calculations
local_proba = ql.LocalProbabilityModel(elink)

# Optional: Set parameters for local detection stages
param_detect = {
    'eta_A': 1.0,
    'eta_B': 1.0,
    'dc_A': 0.0,
    'dc_B': 0.0,
}
local_proba.set_param(param_detect)

# 4. Calculate probabilities
proba_herald = local_proba.get_p_herald()
p00, p01, p10, p11 = local_proba.get_proba()

# 5. Print the results
print("=== Probabilities ===")
print(f"Heralding probability: {proba_herald}")
print(f"Local probabilities:")
print(f"p00 (no-click | no-click) = {p00}")
print(f"p01 (no-click | click)    = {p01}")
print(f"p10 (click | no-click)    = {p10}")
print(f"p11 (click | click)       = {p11}")

# 6. Plot the probabilities against pump values
# fig, ax = local_proba.plot_proba()
# fig.show()
```

## Modules

The library is organized into the following main modules:

-   `src/q_elink/core/link.py`: Contains the core data structure `ElementaryLink` for the quantum elementary link model.
-   `src/q_elink/local_probabilities/model.py`: Contains `LocalProbabilityModel` for calculating detection probabilities.
-   `src/q_elink/witness/model.py`: Provides the `EntanglementWitness` class and related tools for entanglement witnessing.
-   `src/q_elink/local_probabilities/fitter.py`: Contains the `ProbabilityFitter` class for fitting the theoretical model to experimental data.

## Running the Examples

The `examples/` directory contains scripts that demonstrate various features of the library. To run an example, use `poetry run`.

For example, to run the elementary link example:
```bash
poetry run python examples/elem_link.py
```

To run the coincidence measurement example:
```bash
poetry run python examples/coincidence.py
```

## Core Dependencies

Standard python dependencies:
-   [NumPy](https://numpy.org/): For numerical operations.
-   [SciPy](https://scipy.org/): For scientific computing, particularly for data fitting.
-   [Matplotlib](https://matplotlib.org/): For plotting and visualizations.
-   [tqdm](https://github.com/tqdm/tqdm): For progress bars.

