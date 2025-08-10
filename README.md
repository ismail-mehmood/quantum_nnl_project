# Quantum Galton Board Project, with Qubit Stabilisation Extension
Submission for the 2025 WISER Quantum Walks and Monte Carlo project. Created by Ismail Mehmood and Hayden Dyke.

(Please note that the structure of the project has changed slightly, view this an an approximate guide only, use extension.pdf and extension.ipynb to navigate the interferometer project.)

This project implements a quantum simulation of a Galton Box (Plinko board), inspired by the Monte Carlo method. It is part of the Womanium & WISER 2025 Quantum Program.

The simulator demonstrates how quantum circuits can model classical stochastic systems and generate various distributions such as Gaussian, Exponential, and Hadamard-based quantum walks.

---

## Project Structure

```
quantum_nnl_project/
|
├── src/                       # Core source code
│   ├── basic_galton_box.py    # Trial unitary 2D Galton circuit
│   ├── galton_box.py          # Build Galton circuits, simulation utilities
│   ├── exponential_box.py     # Build Galton circuits, simulation utilities for larger model?
│   ├── quantum_walk.py        # Quantum walk circuit
│   ├── analysis.py            # Tools for postprocessing and custom distribution shaping
│   └── noise_optimisation.py  # Apply and simulate noise
│
├── notebooks/                             # Interactive Jupyter notebooks
│   ├── bosonic_galton_box_demo.ipynb      # Boson sampler based demo of a photonic Galton box
│   ├── galton_box_demo.ipynb              # Quantum Galton box examples
│   ├── modified_distribution_boxes.ipynb  # Exponential and geometric versions
│   ├── quantum_walk_demo.ipynb            # Walk demo
│   └── noise_modelling_demo.ipynb         # Hardware noise and mitigations
│
├── results/                   # Output plots and saved data
│   ├── gaussian_histograms/   # TBC, potentially error bars and distribution approximation metrics
│   ├── exponential_outputs/   # TBC, potentially error bars and distribution approximation metrics
│   └── hadamard_walks/        # TBC, potentially error bars and distribution approximation metrics
│
├── docs/                      # Typst documentation and submission deliverables
│   ├── extension/
│   │   └── extension.typ      # Report on additional material, tbc
│   └── summary/
│       ├──  main.typ          # 2-page challenge summary 
│       └──  figures           # Images, bibliography etc
│
├── tests/                     # Unit tests for source modules
│   └── test_galton_box.py     # Some kind of Galton box test, TBC
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

| Notebook                              | Description                                                         |
| ------------------------------------- | ------------------------------------------------------------------- |
| `bosonic_galton_box_demo.ipynb`       | Initial experiment with boson sampling based board.                 |
| `galton_box_demo.ipynb`               | 1-, 2- and N-Layer quantum Galton Board examples.                   |
| `modified_distribution_boxes.ipynb`   | Tweaked circuits for exponential, geometric distribution.           |
| `quantum_walk_demo.ipynb`             | Hadamard and <some other> quantum walk examples.                    |
| `noise_modelling_demo.ipynb`          | Examples with hardware noise, with gate optimisation, ZNE and QEC.  | |

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
f
Note that the notebooks must be run from the ROOT directory of the project, use the os cell at the top of each notebook if you've haven't set this already. 

---

## Testing

Run unit tests with:

```bash
pytest tests/
```

---

## Documentation

All written deliverables (summary, design notes) are in `docs/`. Typst sources are included, and figures are saved in `docs/summary/`.

To compile the PDF (once you have downloaded the Typst compiler):

```bash
cd docs/
typst compile summary/main.typ
```

---
