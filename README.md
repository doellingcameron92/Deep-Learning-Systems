# Deep Learning Systems

CNN image classification on Fashion-MNIST with a controlled experimental comparison (baseline CNN vs. regularized CNN with Batch Normalization and Dropout).

## Contents

- `deep_learning.ipynb` — full workflow: data loading/inspection, baseline CNN, experimental CNN, training, evaluation, comparison, summary.
- `Deep_Learning_Systems_Analysis_Report.pdf` — analysis report with citations.
- `requirements.txt` — generated with `pip freeze > requirements.txt`.
- `data/` — Fashion-MNIST is downloaded automatically by `torchvision.datasets.FashionMNIST` into this folder on first run (see notebook for access instructions).

## Reproduce

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
jupyter nbconvert --to notebook --execute deep_learning.ipynb --output deep_learning.ipynb
```
