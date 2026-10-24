# Not Everyone Has a Voice, or an Ear, on Social Media: The Poster-Observer Model of Opinion Dynamics

This repository contains the anonymized source code to reproduce the algorithms, experimental results, and figures presented in the paper: **"Not Everyone Has a Voice, or an Ear, on Social Media: The Poster-Observer Model of Opinion Dynamics"**.

To enable immediate reproduction without requiring external downloads, two processed benchmark datasets (**Wiki** and **Gnutella**) are included directly within the repository. The larger networks evaluated in the paper (**Epinions**, **Slashdot**, **YouTube**, and **Pokec**) can be downloaded directly from the Stanford Network Analysis Platform (SNAP) at [https://snap.stanford.edu/data/](https://snap.stanford.edu/data/).

---

## 📂 Repository Structure

```text
├── data/
│   ├── raw/                  # Directory for raw .txt.gz graph files downloaded from SNAP
│   └── processed/            # Processed .pt sparse adjacency matrices (Wiki & Gnutella included)
├── results/                  # Generated figures (.pdf) and experiment logs (.csv)
├── src/
│   ├── core.py               # Core PO-FJ dynamics, moment evolutions, and CLEAR-PGD (Alg. 1)
│   ├── graph_loader.py       # Extract Largest Strongly Connected Components (LSCC) and save as .pt
│   ├── rq1.py                # Reproduces RQ1: Behavioral impact heatmaps on Stochastic Block Model
│   ├── rq2.py                # Reproduces RQ2: Conflict mitigation results of CLEAR-PGD and baselines across budgets
│   ├── rq3.py                # Reproduces RQ3: Node-level shift analysis with Pearson correlations
│   ├── rq4.py                # Reproduces RQ4: Scalability and execution runtime scaling results
│   └── appendix.py           # Reproduces Appendix F: K-hop covariance matrix evaluation
├── requirements.txt          # Python environment dependencies
└── README.md
