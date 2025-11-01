# GoodSubspace: Approximate k-Dimensional Subspace Finder (with PCA Baseline)

This repo implements a randomized algorithm for finding a “good” k-dimensional subspace that fits a set of points, compared against PCA as a baseline.  
The code also includes experiments that vary k, ε, number of points, and dimension, plus plots for error and runtime.

## File
- `GoodSubspace.py` — all logic and experiments in one file.

## Method (high level)
- Samples candidate directions biased by vector norms, mixes them via short random “line sequences,” and recursively builds an orthonormal basis for a k-dimensional subspace.
- Measures fit using \( \mathrm{RD}_\tau \) (with \( \tau \in \{1,2,\infty\} \)) and also reports projection error.
- Compares against PCA on the same centered data.

## Dependencies
- Python 3.9+
- numpy
- scipy
- scikit-learn
- matplotlib

Install:
```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -U pip
pip install numpy scipy scikit-learn matplotlib
