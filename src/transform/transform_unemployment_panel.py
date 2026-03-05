import argparse
from pathlib import Path
import pandas as pd

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--infile", default="../../data/raw/unemployment_api.parquet")
    ap.add_argument("--out", default="../../data/processed/unemployment_panel.parquet")
    args = ap.parse_args()

    in_path = Path(args.infile)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(in_path).copy()
    df["period"] = pd.PeriodIndex(df["period"], freq="Q")
    df = df.sort_values(["geo", "period"]).reset_index(drop=True)

    # lag t-1 por país
    df["unemployment_rate_l1"] = df.groupby("geo")["unemployment_rate"].shift(1)

    df.to_parquet(out_path, index=False)

    print("Guardado en:", out_path.resolve())
    print("Geos:", sorted(df["geo"].unique()))
    print(df.groupby("geo").tail(3)[["geo","period","unemployment_rate","unemployment_rate_l1"]].to_string(index=False))

if __name__ == "__main__":
    main()