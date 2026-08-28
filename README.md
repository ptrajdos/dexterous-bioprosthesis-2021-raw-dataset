# Dexterous Bioprosthesis 2021 Raw Datasets

A dataset creation framework for dexterous bioprosthesis research. The package processes raw electromyographic (EMG) and sensor recordings collected during hand gesture experiments, transforming them into structured datasets suitable for machine learning. Instances are represented by raw signal objects that encapsulate multi-channel time-series data.

Key capabilities include:

- **Signal extraction** — extract meaningful features and signal segments from raw EMG recordings using configurable extractors (e.g., delta-RMS).
- **Data augmentation** — expand training datasets through signal-level transformations, including time-warping (DTW/FastDTW), noise injection, and audio-inspired augmentations via `librosa`/`audiomentations`.
- **Dataset creation** — assemble extracted signals into ready-to-use datasets in NumPy and ARFF formats, with support for parallel processing and reproducible pipelines.
- **Hyperparameter optimisation** — tune extraction and augmentation parameters using `hyperopt` and genetic algorithms (`pygad`).
- **Embedded-friendly** — tested on Raspberry Pi Zero with ARM-specific build flags and `numba` JIT compilation support.

## Main Dependencies

| Package | Purpose |
|---|---|
| `numpy` | Array operations and numerical computing |
| `pandas` | Tabular data handling |
| `scipy` | Signal processing and scientific computing |
| `scikit-learn` | Machine learning utilities and preprocessing |
| `matplotlib` | Plotting and visualisation |
| `numba` | JIT compilation for performance-critical code (optional, for augmentation) |
| `dtw-python` / `fastdtw` | Dynamic Time Warping for signal alignment |
| `sktime` | Time-series analysis toolkit |
| `PyWavelets` | Wavelet transforms for signal decomposition |
| `statsmodels` | Statistical modelling and tests |
| `pygad` | Genetic algorithm optimisation |
| `hyperopt` | Hyperparameter optimisation |
| `csaps` | Cubic smoothing splines |
| `kneed` | Knee/elbow point detection in curves |
| `liac-arff` | ARFF file format support |
| `joblib` | Parallel execution and caching |
| `tqdm` | Progress bars |
| `Cython` | C-extension compilation |
| `librosa` | Audio/signal augmentation (optional, augmentation extra) |
| `audiomentations` | Audio augmentation transforms (optional, augmentation extra) |

## Installation

### Via pip (local)

```bash
pip install -e .
```

To include optional augmentation dependencies:

```bash
pip install -e ".[augmentation]"
```

### Via pip (git repository)

Install directly from the GitHub repository:

```bash
pip install git+https://github.com/ptrajdos/dexterous-bioprosthesis-2021-raw-dataset.git
```

To include optional augmentation dependencies:

```bash
pip install "dexterous_bioprosthesis_2021_raw_datasets[augmentation] @ git+https://github.com/ptrajdos/dexterous-bioprosthesis-2021-raw-dataset.git"
```

To install a specific branch or tag:

```bash
pip install git+https://github.com/ptrajdos/dexterous-bioprosthesis-2021-raw-dataset.git@branch-name
```

### Via Makefile

The Makefile automates environment setup, testing, documentation, and static analysis. All targets that require Python packages will automatically create a virtual environment and install dependencies.

#### Quick Start

```bash
make pypackages
```

This creates a virtual environment, installs the package in editable mode along with development dependencies from `requirements_dev.txt`.

#### Makefile Targets

| Target | Description |
|---|---|
| `make all` | Default target. Runs profiling tests with coverage. |
| `make venv` | Creates a Python virtual environment in `./venv` and installs basic build tools (`wheel`, `setuptools`). |
| `make pypackages` | Creates the virtual environment (if needed) and installs the package in editable mode with all development dependencies. |
| `make test` | Runs the full test suite using `unittest` with branch coverage. Generates an HTML coverage report. Requires test data to be unpacked. |
| `make test_parallel` | Runs tests in parallel using `unittest-parallel` with coverage. |
| `make profile` | Runs tests with `pytest` using profiling and coverage reporting. |
| `make docs` | Generates HTML documentation using `pdoc3` and UML diagrams. |
| `make sphinx` | Generates HTML documentation using Sphinx with autodoc, UML integration via `sphinx-pyreverse`, and Read the Docs theme. Output in `docs_sphinx/_build`. |
| `make uml` | Generates UML class and package diagrams (SVG) using `pyreverse` and `graphviz`. |
| `make static_check` | Runs all static analysis tools (`flake8`, `mypy`, `pylint`). |
| `make flake8` | Runs `flake8` linter. Output saved to `static_analysis/flake8.log`. |
| `make mypy` | Runs `mypy` type checker. Output saved to `static_analysis/mypy.log`. |
| `make lint` | Runs `pylint`. Output saved to `static_analysis/lint.json`. |
| `make tox_check` | Runs tests across multiple Python versions using `tox`. |
| `make data_unp` | Unpacks test data archive from `data/` directory. |
| `make clean` | Removes build artifacts, virtual environment, tox cache, and test logs. |
| `make clean_venv` | Removes only the virtual environment and logs. |
| `make clean_pypackages` | Removes the `pypackages` marker file. |
| `make clean_tox` | Removes the `.tox` directory. |

## Troubleshooting

### Missing `libffi`

If you encounter errors related to `libffi` (e.g., `ModuleNotFoundError: No module named '_ctypes'`), you need to build and install it from source. A helper script is provided:

```bash
bash install_libffi.sh
```

The script automatically selects the appropriate `libffi` version based on your Python version (3.3 for Python 3.9, 3.5.2 for Python 3.11), downloads, compiles, and installs it. After installation, you may need to rebuild Python for it to pick up the new `libffi`.

### Raspberry Pi Zero flags

When building native dependencies on a Raspberry Pi Zero (ARMv6/ARMv7), you may need to set specific compiler flags for optimal performance. Source the provided script before installing:

```bash
source pizero_flags.sh
pip install -e .
```

This sets the following environment variables:

```bash
export CFLAGS="-mcpu=cortex-a53 -mfpu=neon-fp-armv8 -O3 -ftree-vectorize"
export CC=gcc
export CXXFLAGS="$CFLAGS"
```
