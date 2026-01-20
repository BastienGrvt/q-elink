# Project Overview

This project, `q-elink`, is a Python library for the modelization of quantum elementary links. It provides tools to simulate, analyze, and fit experimental data related to quantum entanglement and quantum communication. The core of the library is built around the concept of an "elementary link," which is a fundamental building block in quantum networks.

The library is written in Python and utilizes several scientific computing libraries:
- **NumPy:** For numerical operations and array manipulations.
- **SciPy:** For scientific and technical computing, specifically for fitting experimental data.
- **Matplotlib:** For creating static, animated, and interactive visualizations.
- **tqdm:** For displaying progress bars.
- **rich:** For rich text and beautiful formatting in the terminal.

The project is structured as a Python package and uses [Poetry](https://python-poetry.org/) for dependency management and packaging.

## Building and Running

This project is a library, so it's not meant to be "run" in the traditional sense. Instead, it's intended to be imported and used in other Python scripts or Jupyter notebooks.

### Installation

To install the necessary dependencies, you can use Poetry:

```bash
poetry install
```

To run a script (for isntance `elem_link.py`):

```bash
poetry run python examples/...
```


### Usage

The library can be used by importing the necessary classes and functions from the `q_elink` package. For example, to use the `ElementaryLink` class:

```python
from q_elink import ElementaryLink

# Create an instance of the ElementaryLink class
elink = ElementaryLink()

# Set the parameters
elink.set_param({
    "p_A": 0.5,
    "p_B": 0.5,
    "eta_0": 0.8,
    "eta_A": 0.9,
    "eta_B": 0.9,
    "dc_0": 1e-6,
    "dc_A": 1e-6,
    "dc_B": 1e-6,
})

# ...
```

The `examples` directory contains scripts that demonstrate how to use the library.

## Development Conventions

The codebase is organized into several modules, each with a specific purpose:

- **`elem_link_model.py`**: Defines the core data structures and models for the quantum elementary link.
- **`ent_witness.py`**: Provides tools for entanglement witnessing.
- **`fit_data.py`**: Contains functionality for fitting the theoretical model to experimental data.

The code is well-documented with comments and docstrings, and it uses type hints to improve code clarity and maintainability. The project follows standard Python coding conventions (PEP 8).
