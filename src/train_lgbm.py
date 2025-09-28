# LightGBM model with iterative 16-day forecasting.
from __future__ import annotations
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_log_error
import lightgbm as lgb
from tqdm import tqdm

RAW = Path("data/raw")
SUB_DIR = Path("data/submissions"); SUB_DIR.mkdir(parents=True, exist_ok=True)

FORECAST_DAYS = 16  # 2017-08-16 .. 2017-08-31

def rmsle(y_true, y_pred) -> float:
    from sklearn.metrics import mean_squared_log_error
    import numpy as np
    # RMSLE = sqrt(MSLE)
    return np.sqrt(mean_squared_log_error(y_true, y_pred))
        
def add_calendar_feats(df: pd.DataFrame) -> pd.DataFrame:
    d = df["date"]
    return df.assign(
        dow=d.dt.dayofweek.astype(np.int16),
        dom=d.dt.day.astype(np.int16),
        week=d.dt.isocalendar().week.astype(np.int16),
        month=d.dt.month.astype(np.int16),
        year=d.dt.year.astype(np.int16),
        is_weekend=(d.dt.dayofweek>=5).astype(np.int8),
        is_month_end=d.dt.is_month_end.astype(np.int8),
        is_month_start=d.dt.is_month_start.astype(np.int8),
        is_payday=((d.dt.day==15)|(d.dt.is_month_end)).astype(np.int8),
    )

def make_lags(df: pd.DataFrame, group_cols: list[str], target_col: str="sales") -> pd.DataFrame:
    df = df.sort_values(["store_nbr","family","date"]).copy()
    for lag in [1, 7, 14, 28]:
        df[f"lag_{lag}"] = df.groupby(group_cols)[target_col].shift(lag)
    for w in [7, 14, 28]:
        df[f"rmean_{w}"] = (
            df.groupby(group_cols)[target_col]
              .transform(lambda s: s.shift(1).rolling(w, min_periods=1).mean())
        )
    return df

def prepare_history(train: pd.DataFrame) -> pd.DataFrame:
    hist = train.copy()
    hist = add_calendar_feats(hist)
    return hist

def fit_encoder(series: pd.Series) -> LabelEncoder:
    le = LabelEncoder()
    le.fit(series.astype(str))
    return le

def train_val_split(history: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    last_train_date = history["date"].max()
    val_start = last_train_date - pd.Timedelta(days=15)
    train_df = history[history["date"] < val_start].copy()
    val_df   = history[history["date"] >= val_start].copy()
    return train_df, val_df

def build_train_matrix(train_df: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
    feats = [
        "store_nbr","family_enc",
        "dow","dom","week","month","year","is_weekend","is_month_end","is_month_start","is_payday",
        "onpromotion",
        "lag_1","lag_7","lag_14","lag_28",
        "rmean_7","rmean_14","rmean_28",
    ]
    X = train_df[feats].copy()
    y = np.log1p(train_df["sales"].values)
    return X, y, feats

def train_model(train_df: pd.DataFrame, val_df: pd.DataFrame):
    fam_le = fit_encoder(pd.concat([train_df["family"], val_df["family"]]))
    for df in (train_df, val_df):
        df["family_enc"] = fam_le.transform(df["family"].astype(str))

    X_tr, y_tr, feats = build_train_matrix(train_df)
    X_va, y_va, _     = build_train_matrix(val_df)

    model = lgb.LGBMRegressor(
        n_estimators=3000,
        learning_rate=0.05,
        num_leaves=255,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=0.1,
        random_state=42,
    )
    model.fit(
        X_tr, y_tr,
        eval_set=[(X_va, y_va)],
        eval_metric="rmse",
        callbacks=[lgb.early_stopping(100)]
    )

    val_pred = np.expm1(model.predict(X_va, num_iteration=model.best_iteration_)).clip(0)
    val_score = rmsle(val_df["sales"].values, val_pred)
    print(f"Validation RMSLE: {val_score:.5f}")
    return model, fam_le, feats

def iterative_forecast(model, fam_le, feats, history: pd.DataFrame, test: pd.DataFrame) -> pd.DataFrame:
    future_dates = sorted(test["date"].unique())
    history = history.copy()
    preds = []

    for dt in tqdm(future_dates, desc="Iterating days"):
        today_rows = test[test["date"] == dt].copy()
        tmp = pd.concat([history, today_rows.assign(sales=np.nan)], ignore_index=True)

        tmp = add_calendar_feats(tmp)
        tmp = make_lags(tmp, group_cols=["store_nbr","family"], target_col="sales")

        today_feats = tmp[tmp["date"] == dt].copy()
        today_feats["family_enc"] = fam_le.transform(today_feats["family"].astype(str))

        yhat = np.expm1(model.predict(today_feats[feats], num_iteration=getattr(model, "best_iteration_", None))).clip(0)
        today_rows = today_rows.copy()
        today_rows["sales"] = yhat

        history = pd.concat([history, today_rows], ignore_index=True)
        preds.append(today_rows[["id","sales"]])

    return pd.concat(preds, ignore_index=True)

def main():
    train = pd.read_csv(RAW / "train.csv", parse_dates=["date"])
    test  = pd.read_csv(RAW / "test.csv",  parse_dates=["date"])

    use_cols = ["date","store_nbr","family","onpromotion","sales"]
    train = train[use_cols].copy()
    test  = test[["id","date","store_nbr","family","onpromotion"]].copy()

    history = prepare_history(train)
    history = make_lags(history, group_cols=["store_nbr","family"], target_col="sales")

    history = history.dropna(subset=["lag_1","lag_7","lag_14","lag_28"])

    tr_df, va_df = train_val_split(history)

    model, fam_le, feats = train_model(tr_df, va_df)

    preds = iterative_forecast(model, fam_le, feats, history, test)

    out_path = SUB_DIR / "submission_lgbm_iterative.csv"
    preds.to_csv(out_path, index=False)
    print(f"Wrote {out_path} ({len(preds):,} rows)")

if __name__ == "__main__":
    main()