# Super-simple baseline:
# For each (store_nbr, family), forecast the mean of the last 16 days of sales in train.
import pandas as pd
from pathlib import Path

RAW = Path("data/raw")
SUB_DIR = Path("data/submissions"); SUB_DIR.mkdir(parents=True, exist_ok=True)

def main():
    train = pd.read_csv(RAW / "train.csv", parse_dates=["date"])
    test  = pd.read_csv(RAW / "test.csv",  parse_dates=["date"])
    ss    = pd.read_csv(RAW / "sample_submission.csv")

    last_day = train["date"].max()
    last16_start = last_day - pd.Timedelta(days=15)  # inclusive
    recent = train.loc[train["date"].between(last16_start, last_day)]

    avg16 = (
        recent.groupby(["store_nbr", "family"], as_index=False)["sales"]
              .mean().rename(columns={"sales": "pred"})
    )

    overall = train["sales"].mean()
    out = (test[["id","store_nbr","family"]]
           .merge(avg16, on=["store_nbr","family"], how="left"))
    out["sales"] = out["pred"].fillna(overall).clip(lower=0)
    out = out[["id","sales"]]

    path = SUB_DIR / "submission_baseline_last16.csv"
    out.to_csv(path, index=False)
    print(f"Wrote {path} ({len(out):,} rows)")

if __name__ == "__main__":
    main()