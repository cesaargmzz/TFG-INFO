import argparse
from pathlib import Path
import pandas as pd

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--infile", default="../../data/raw/gdp_eurostat_api.parquet")
    ap.add_argument("--out", default="../../data/processed/gdp_qoq_panel.parquet")
    args = ap.parse_args()

    in_path = Path(args.infile)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(in_path).copy()

    # Normalizamos nombres
    df = df.rename(columns={"value": "gdp_level"})

    # Asegurar orden
    df = df.sort_values(["geo", "period"])

    # QoQ %
    df["gdp_qoq_pct"] = df.groupby("geo")["gdp_level"].pct_change() * 100

    # Lags del target por país
    df["gdp_qoq_pct_l1"] = df.groupby("geo")["gdp_qoq_pct"].shift(1)
    df["gdp_qoq_pct_l2"] = df.groupby("geo")["gdp_qoq_pct"].shift(2)

    df.to_parquet(out_path, index=False)

    print("Guardado en:", out_path.resolve())
    print("Geos:", sorted(df["geo"].unique()))
    print(df.groupby("geo").tail(3)[["geo", "period", "gdp_level", "gdp_qoq_pct", "gdp_qoq_pct_l1", "gdp_qoq_pct_l2"]])

if __name__ == "__main__":
    main()