# Not Everyone Has a Voice, or an Ear, on Social Media: The Poster-Observer Model of Opinion Dynamics

This repository contains the anonymized source code to reproduce the algorithms, experimental results, and figures presented in the paper: **"Not Everyone Has a Voice, or an Ear, on Social Media: The Poster-Observer Model of Opinion Dynamics"**.

To enable immediate reproduction without requiring external downloads, two processed benchmark datasets (**Wiki** and **Gnutella**) are included directly within the repository. The larger networks evaluated in the paper (**Epinions**, **Slashdot**, **YouTube**, and **Pokec**) can be downloaded directly from the Stanford Network Analysis Platform (SNAP) at [https://snap.stanford.edu/data/](https://snap.stanford.edu/data/).

---

## 📂 Repository Structure

```text
├── data/                     # Graph datasets (Wiki & Gnutella included as .pt and .txt)
├── results/                  # Generated figures (.pdf) and experiment logs (.csv)
├── src/
│   ├── po_fj_dynamics.py     # PO-FJ expected W, moment evolutions, and conflict metrics
│   ├── clear_optimizer.py    # CLEAR-PGD optimizer with checkpointing and budget projection
│   ├── baselines.py          # Random, Ideological-Heuristic, and Bridge-Heuristic baselines
│   ├── graph_utils.py        # Spectral bisection and 90-9-1 user scenario generator
│   ├── graph_loader.py       # SNAP edgelist processor (LSCC extraction -> .pt export)
│   ├── run_main.py           # Main benchmark driver executing CLEAR-PGD and baselines
│   ├── rq1.py                # Reproduces RQ1: Behavioral impact heatmaps on SBM
│   ├── rq2.py                # Reproduces RQ2: Conflict mitigation plots across budgets
│   ├── rq3.py                # Reproduces RQ3: Node-level attribution and Pearson correlations
│   └── rq4.py                # Reproduces RQ4: Scalability runtime benchmark and log-log scaling plot
├── requirements.txt          # Python environment dependencies
├── LICENSE                   # MIT License
└── README.md
```

---

## Environment Setup

This codebase was developed and evaluated using Python 3.10+. To set up the environment, install the required packages:

```bash
pip install -r requirements.txt
