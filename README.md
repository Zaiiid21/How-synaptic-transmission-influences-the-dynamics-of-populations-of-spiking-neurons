# How Synaptic Transmission Influences the Dynamics of Populations of Spiking Neurons

Code repository for a Bachelor's Thesis (TFG) on spiking-neuron population dynamics, comparing:

- `CUBA`: current-based synaptic coupling
- `COBA`: conductance-based synaptic coupling
- Single-population and excitatory-inhibitory (E-I) population settings
- Full network simulations vs reduced mean-field ODE systems

The implementation is mostly in Python, with MATLAB live scripts (`.mlx`) used for post-processing and bifurcation/phase-diagram analyses.

## Repository Structure

- `COBA_single_neuron.py`
  - Single-neuron COBA test and visualization.
- `HH_code.py`
  - Hodgkin-Huxley reference implementation and comparison plot.
- `Potencial_de_acción.py`
  - Action potential shape generation/plotting.
- `graph_neurons.py`
  - Draws an all-to-all neuron graph using `networkx` and a neuron image.

- `Codigo Resultados COBA/`
  - Main single-population COBA simulation (`COBA.py`).
  - MATLAB scripts/notebooks for COBA result analysis.

- `CUBA/`
  - Main single-population CUBA simulation (`CUBA.py`).
  - Parameter presets and MATLAB notebooks:
    - `Focus, Node and Saddle/`
    - `Sinusoidal/`

- `CUBA non-coupled/`
  - CUBA non-coupled/synchrony comparison script (`non-coupled CUBA.py`).
  - Saved `.mat` outputs for synchronized/non-synchronized cases.

- `Código señales CUBA2 vs COBA2/`
  - Two-population E-I models:
    - `COBA EI/COBA 2.py`
    - `CUBA EI/CUBA 2.py`
  - Each folder includes a parameter text file and MATLAB live script.

- `Pruebas para COBAvsCUBA EDOS/`
  - Reduced ODE-only test scripts:
    - `COBAEDOSpy.py`, `CUBAEDOS.py`
    - `COBA2EDOS.py`, `CUBA2EDOS.py`

- `Codigo Resultados COBA2 CUBA2/`
  - MATLAB live scripts, saved `.mat` datasets, and phase-diagram figures for COBA2/CUBA2 studies.

## Requirements

Python 3.10+ is recommended.

Install dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install numpy scipy matplotlib pandas networkx
```

Notes:

- Most simulation scripts use `numpy`, `scipy`, and `matplotlib`.
- `Potencial_de_acción.py` uses `pandas`.
- `graph_neurons.py` uses `networkx`.

## Quick Start

Run any script directly (parameters are defined inside each file):

```powershell
python "Codigo Resultados COBA\COBA.py"
python "CUBA\CUBA.py"
python "CUBA non-coupled\non-coupled CUBA.py"
python "Código señales CUBA2 vs COBA2\COBA EI\COBA 2.py"
python "Código señales CUBA2 vs COBA2\CUBA EI\CUBA 2.py"
```

For reduced ODE tests:

```powershell
python "Pruebas para COBAvsCUBA EDOS\COBAEDOSpy.py"
python "Pruebas para COBAvsCUBA EDOS\CUBAEDOS.py"
python "Pruebas para COBAvsCUBA EDOS\COBA2EDOS.py"
python "Pruebas para COBAvsCUBA EDOS\CUBA2EDOS.py"
```

## Outputs

Depending on the script, outputs include:

- Time-series plots (membrane voltage and firing rates)
- Raster-like activity matrices
- MATLAB `.mat` files for downstream analysis (e.g., `V_*.mat`, `FR*.mat`, `Raster*.mat`)
- Phase/bifurcation assets (mainly in MATLAB workflow folders)

## Practical Notes

- Many scripts default to large populations (`N` up to 10,000), which can be slow. For quick tests, reduce:
  - `N`
  - `Tmax`
  - Increase `dt` cautiously
- Scripts are designed as standalone experiment files, not as a packaged module.
- `graph_neurons.py` contains a hardcoded absolute path to a neuron image:
  - `C:\Users\zaidh\Desktop\TFG\Figures\neuron.png`
  - Update this path before running.

## MATLAB Workflow

MATLAB live scripts (`.mlx`) are included for:

- Post-processing simulation outputs
- Preparing phase diagrams
- Hopf/bifurcation-related visual analysis

Primary MATLAB analysis folders:

- `Codigo Resultados COBA/`
- `Codigo Resultados COBA2 CUBA2/`
- `CUBA/Focus, Node and Saddle/`
- `CUBA/Sinusoidal/`
- `Código señales CUBA2 vs COBA2/*`

## Context

This repository supports the thesis project on how synaptic transmission rules (current-based vs conductance-based) affect collective dynamics in populations of spiking neurons.
