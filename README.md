# Store Sales — Time Series Forecasting

Forecast daily **unit sales** for each **(store_nbr, family)** pair for the 16‑day test window (2017‑08‑16 → 2017‑08‑31) using classic time‑series features and gradient boosting. This repository is tailored for the Kaggle competition:  
https://www.kaggle.com/competitions/store-sales-time-series-forecasting

> **Metric:** Root Mean Squared Logarithmic Error (**RMSLE**), evaluated on `sales` (non‑negative, continuous).  
> **Data:** Corporación Favorita (Ecuador) aggregated at *store × product family*, with auxiliary tables (`oil`, `holidays_events`, `transactions`, `stores`) available.

---

## 🚀 Quickstart

```bash
# 0) Python (>=3.10 recommended)
python --version

# 1) Create & activate a virtual environment
python -m venv .venv && source .venv/bin/activate   # (macOS/Linux)
# .venv\Scripts\activate                             # (Windows PowerShell)

# 2) Install dependencies
pip install -U pip
pip install pandas numpy scikit-learn lightgbm tqdm kaggle

# 3) Configure Kaggle API (one‑time)
# Place kaggle.json (Account → Create New API Token) at:
#   macOS/Linux: ~/.kaggle/kaggle.json
#   Windows:     %USERPROFILE%\.kaggle\kaggle.json
chmod 600 ~/.kaggle/kaggle.json  # (macOS/Linux)

# 4) Download & unpack competition data into data/raw/
python download_data.py

# 5) Run a super‑simple baseline (mean of last 16 days per series)
python baseline_last16.py

# 6) Train LightGBM model and generate an iterative 16‑day forecast
python train_lgbm.py

# 7) Submit to Kaggle (choose one of the generated CSVs)
kaggle competitions submit -c store-sales-time-series-forecasting \
  -f data/submissions/submission_lgbm_iterative.csv -m "LGBM iterative v1"
```

---

## 📁 Project Structure

```
.
├── baseline_last16.py           # Mean of last 16 days per (store, family) with fallback
├── train_lgbm.py                # LightGBM regressor with iterative day-by-day forecasting
├── download_data.py             # Uses Kaggle CLI to fetch & unzip data
├── data/
│   ├── raw/                     # All CSVs from the competition live here
│   │   ├── train.csv
│   │   ├── test.csv
│   │   ├── sample_submission.csv
│   │   ├── oil.csv
│   │   ├── holidays_events.csv
│   │   ├── transactions.csv
│   │   └── stores.csv
│   └── submissions/             # Generated Kaggle submissions (*.csv)
└── README.md
```

---

## 🧠 Approaches

### 1) **Baseline — “Last‑16 Mean”** (`baseline_last16.py`)
- For each `(store_nbr, family)`, compute the **mean of the last 16 days** in `train.csv` and use it for all test dates.
- If a series is missing in train (or insufficient history), **fallback** to the **global mean** of `sales`.
- Clip negatives to 0 and write: `data/submissions/submission_baseline_last16.csv`.

This is intentionally simple but surprisingly competitive as a sanity check and integration test.

---

### 2) **LightGBM — Iterative Forecast** (`train_lgbm.py`)
A gradient‑boosted decision tree regressor trained on time‑series features with **log1p target** and **day‑by‑day roll‑forward prediction** for the 16‑step horizon.

Key details (skim the script for the exact implementation):
- **Target & metric:** model is trained on `log1p(sales)` and validated with **RMSLE**.
- **Categoricals:** label‑encode `family` (and other categoricals if present).
- **Lag features:** uses classic **lags** such as **1, 7, 14, 28** days (rows with missing lags are dropped for training).
- **Calendar parts:** simple date parts (e.g., day of week / month) are typically useful; extend as you wish.
- **Iterative 16‑day loop:** predict day `t+1`, append to history, recompute features, then predict `t+2`, … until `t+16`.
- **Validation:** a **time‑based split** near the end of the training range; the script prints `Validation RMSLE: ...`.
- **Output:** `data/submissions/submission_lgbm_iterative.csv`.

> Tip: You can extend features by merging `oil.csv`, `holidays_events.csv`, `transactions.csv`, and `stores.csv` on `date`/`store_nbr` keys to capture additional signal (e.g., macro trends, holiday effects, traffic).

---

## 🔬 Reproducing Results

1. Ensure data exists under `data/raw/` (`python download_data.py`).
2. Run `python baseline_last16.py` to sanity‑check I/O and pathing.
3. Run `python train_lgbm.py` and note the **validation RMSLE** printed.
4. Submit the produced file to verify **LB score** vs. **CV**.

> **Expected behavior:** CV and LB should be directionally aligned; if they drift, revisit the time split and feature leakage (e.g., using test‑period info inadvertently).

---

## ⚙️ Configuration & Dependencies

Minimal set (pin as needed for reproducibility):
```
python>=3.10
pandas>=2.0
numpy>=1.24
scikit-learn>=1.2
lightgbm>=4.0
tqdm>=4.65
kaggle>=1.5
```

**macOS note (LightGBM):**
If you hit OpenMP errors, install the runtime:
```bash
brew install libomp
```

---

## 🧪 Validation Strategy (suggested)

Time‑series CV that **respects chronology**. A simple, reliable scheme is:
- **Train:** up to 2017‑08‑15 (inclusive)
- **Valid:** a recent contiguous window (e.g., last 16–28 days of `train`)
- **Test:** 2017‑08‑16 .. 2017‑08‑31 (provided by Kaggle)

You can also implement **roll‑forward CV folds** (e.g., 3 folds with progressively later validation windows) for more robust model selection.

---

## 📈 Ideas & Next Steps

- Add **rolling means/stds** over 7/14/28 windows (by series and/or overall).
- Merge **oil / holidays / transactions / stores**; engineer domain features (holiday proximity, oil shocks).
- Add **promotions** dynamics (e.g., recent promo intensity per series).
- Try **CatBoost** / **XGBoost** / **NGBoost** baselines.
- Train **one‑step models** for each horizon (16 separate models) vs. a single iterative model.
- Explore **global models** (single model across all series) vs. per‑family models.
- Consider **neural** baselines (LSTM/GRU/Temporal‑Fusion‑Transformer) if you want to go deep learning.
- Blend strong but diverse models to stabilize LB.

---

## 📝 Submissions

Generated files land in `data/submissions/` with self‑describing names, e.g.:
- `submission_baseline_last16.csv`
- `submission_lgbm_iterative.csv`

Conventional Kaggle CLI submission:
```bash
kaggle competitions submit -c store-sales-time-series-forecasting \
  -f data/submissions/submission_baseline_last16.csv -m "baseline last16 mean"
```

---

## 🤝 Acknowledgements

- Data & problem by **Kaggle** and **Corporación Favorita** (Ecuador).  
- The community’s public notebooks were invaluable for inspiration.

---

## 📜 License

This project is provided for educational purposes. 
