# Store Sales — Time Series Forecasting (Kaggle)

End‑to‑end, GitHub‑ready project for the Kaggle competition: *Store Sales — Time Series Forecasting*.
It includes a dead‑simple baseline and a stronger LightGBM model with an iterative 16‑day forecast loop.

> Competition: https://www.kaggle.com/competitions/store-sales-time-series-forecasting

## 0) Quickstart (local)

```bash
# 0. Clone this repo (or unzip the starter and `cd` into it)
cd store-sales-kaggle

# 1. Python env (any method is fine)
python -m venv .venv && source .venv/bin/activate    # macOS/Linux
# or: py -m venv .venv && .venv\Scripts\activate    # Windows

# 2. Install deps
pip install -r requirements.txt

# 3. (one‑time) Kaggle API setup
#    - Create kaggle token at https://www.kaggle.com/settings/account  (Create New API Token)
#    - Move the downloaded `kaggle.json` to ~/.kaggle/kaggle.json and chmod 600 on Unix
mkdir -p ~/.kaggle && mv ~/Downloads/kaggle.json ~/.kaggle/kaggle.json
chmod 600 ~/.kaggle/kaggle.json  # macOS/Linux

# 4. Download data
python src/download_data.py

# 5. Make a baseline submission (last-16-day store&family average)
python src/baseline_last16.py

# 6. Train LightGBM and create submission (iterative 16-day loop)
python src/train_lgbm.py
```

Submissions will be written to `data/submissions/`.

## Project layout

```
store-sales-kaggle/
├─ data/
│  ├─ raw/           # Kaggle CSVs land here (train.csv, test.csv, stores.csv, ...)
│  ├─ processed/     # Any intermediate feature tables (optional)
│  └─ submissions/   # CSVs ready to upload to Kaggle
├─ notebooks/        # EDA notebooks (add your own)
├─ src/
│  ├─ download_data.py      # uses Kaggle API
│  ├─ baseline_last16.py    # super-simple baseline
│  └─ train_lgbm.py         # LightGBM with iterative forecasting
├─ requirements.txt
└─ .gitignore
```

## Notes & tips

- Metric is **RMSLE**; training on `log1p(sales)` and evaluating with RMSLE usually aligns well.
- Test window is **2017‑08‑16 → 2017‑08‑31** (16 days). We'll mirror that in validation.  
- Start with the baseline to confirm your pipeline end‑to‑end, then iterate on features and model.
- Useful features to add later: more lags/rolls, holiday flags, oil price, pay‑day flags (15th/last day), city/state/store type, month/week/dow, etc.

## How to push to GitHub

```bash
git init
git add .
git commit -m "Initial commit: Kaggle Store Sales starter"
git branch -M main
git remote add origin https://github.com/<your-username>/store-sales-kaggle.git
git push -u origin main
```