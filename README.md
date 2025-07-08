# quantum_nnl_project
Submission for the 2025 WISER Quantum Walks and Monte Carlo project, sponsored by NNL. Created by Ismail Mehmood and Hayden Dyke.

This project implements a quantum simulation of a Galton Box (Plinko board), inspired by the Monte Carlo method. It is part of the Womanium & WISER 2025 Quantum Program in collaboration with NNL.

The simulator demonstrates how quantum circuits can model classical stochastic systems and generate various distributions such as Gaussian, Exponential, and Hadamard-based quantum walks.

---

## Project Structure

```
quantum_nnl_project/
|
├── src/                       # Core source code (reusable functions & modules)
│   ├── basic_galton_box.py    # Trial unitary 2D Galton circuit
│   ├── galton_box.py          # Build Galton circuits, simulation utilities
│   ├── exponential_box.py     # Build Galton circuits, simulation utilities for larger model?
│   ├── quantum_walk.py        # Quantum walk circuit
│   ├── analysis.py            # Tools for postprocessing and custom distribution shaping
│   └── noise_optimisation.py  # Apply and simulate noise
│
├── notebooks/                           # Interactive Jupyter notebooks
│   ├── basic_galton_box.ipynb           # Initial Gaussian simulation
│   ├── galton_box_demo.ipynb            # Parametric version & visualizations
│   ├── exponential_box_demo.ipynb       # Exponential version & visualizations
│   ├── noise_modelling_demo.ipynb       # Noisy demo
│   └── quantum_walk_demo.ipynb          # Walk demo
│
├── results/                   # Output plots and saved data
│   ├── gaussian_histograms/ # TBC
│   ├── exponential_outputs/ # TBC
│   └── hadamard_walks/      # TBC
│
├── docs/                      # LaTeX documentation and submission deliverables
│   ├── notes/
│   │   └── formulae.tex          # Technical/design notes
│   ├── summary/
│   │   └── main.tex           # 2-page challenge summary 
│   └── figures/
│
├── tests/                     # Unit tests for source modules
│   └── test_galton_box.py     # Some kind of Galton box test
│
├── .gitignore
├── README.md
└── requirements.txt           # Python dependencies
```

---

## TODO: 

*  Build quantum Galton Box circuits with configurable layers
*  Generate and visualize Gaussian, Exponential, and Quantum Walk distributions
*  Simulate noise using real-device models (e.g., IBM backends)
*  Compute statistical distances (TVD, KL divergence) vs. target distributions

---

## Notebooks Overview

| Notebook                         | Description                                         |
| -------------------------------- | --------------------------------------------------- |
| `galton_box_basic.ipynb`         | First prototype: Gaussian from quantum randomness   |
| `galton_box_demo.ipynb`          | Generalized circuit with any number of layers       |
| `distribution_experiments.ipynb` | Tweaked circuits for exponential and Hadamard walks |

---

## How to Run

1. Clone the repository:

   ```bash
   git clone https://github.com/ismail-mehmood/quantum-nnl-project.git
   cd quantum-nnl-project
   ```

2. Create a virtual environment and install dependencies:

   ```bash
   python -m venv venv
   source venv/bin/activate   # or `venv\Scripts\activate` on Windows
   pip install -r requirements.txt
   ```

3. Launch notebooks:

   ```bash
   jupyter notebook
   ```

Note that the notebooks must be run from the ROOT directory of the project, if using the integrated notebook editor in VSCode, please adjust the .vscode config accordingly.

---

## Testing

Run unit tests with:

```bash
pytest tests/
```

---

## Documentation

All written deliverables (summary, design notes) are in `docs/`. LaTeX sources are included, and figures are saved in `docs/figures/`.

To compile the PDF:

```bash
cd docs/
pdflatex report.tex
```

---
