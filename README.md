# Deep Learning Systems

CNN image classification on Fashion-MNIST with a controlled experimental comparison (baseline CNN vs. regularized CNN with Batch Normalization and Dropout).

## Contents

- `deep_learning.ipynb` — full workflow: data loading/inspection, baseline CNN, experimental CNN, training, evaluation, comparison, summary.
- `Deep_Learning_Systems_Analysis_Report.pdf` — analysis report with citations.
- `requirements.txt` — generated with `pip freeze > requirements.txt` (CPU-only PyTorch build).
- `artifacts/` — metrics CSVs and figures exported by the notebook and used by the report.
- `build_notebook.py`, `build_report.py` — source scripts that generate the notebook and the PDF report.
- `data/` — Fashion-MNIST is downloaded automatically by `torchvision.datasets.FashionMNIST` into this folder on first run (see notebook for access instructions).

## Results (seed 42, 12 epochs, Adam 1e-3, batch 128)

| | Baseline CNN | Regularized CNN (BN + Dropout) |
|---|---|---|
| Test accuracy | 0.9176 | 0.9228 |
| Test macro-F1 | 0.9169 | 0.9221 |
| Test loss | 0.254 | 0.218 |
| Train–test accuracy gap | 4.4 pp | 0.4 pp |

## Reproduce

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install --extra-index-url https://download.pytorch.org/whl/cpu -r requirements.txt
jupyter nbconvert --to notebook --execute deep_learning.ipynb --output deep_learning.ipynb   # ~4 min on 8 CPU cores
python build_report.py   # regenerates the PDF from artifacts/
```
