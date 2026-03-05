import argparse
from pathlib import Path
import pandas as pd

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--infile", default="../../data/raw/inflation_api.parquet")
    ap.add_argument("--out", default="../../data/processed/inflation_qoq_panel.parquet")
    args = ap.parse_args()

    in_path = Path(args.infile)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(in_path).copy()

    # asegurar period mensual
    df["period"] = pd.PeriodIndex(df["period"], freq="M")

    # ordenar para seguridad
    df = df.sort_values(["geo", "period"]).reset_index(drop=True)

    # 1) agregación mensual -> media trimestral
    df["quarter"] = df["period"].dt.to_timestamp().dt.to_period("Q")
    quarterly = (
        df.groupby(["geo", "quarter"], as_index=False)["hicp_index"]
        .mean()
        .rename(columns={"quarter": "period"})  # ahora period es trimestral
    )

    quarterly = quarterly.sort_values(["geo", "period"]).reset_index(drop=True)

    # 2) QoQ %
    quarterly["inflation_qoq_pct"] = quarterly.groupby("geo")["hicp_index"].pct_change() * 100

    # 3) lag t-1
    quarterly["inflation_qoq_pct_l1"] = quarterly.groupby("geo")["inflation_qoq_pct"].shift(1)

    quarterly.to_parquet(out_path, index=False)

    print("Guardado en:", out_path.resolve())
    print("Geos:", sorted(quarterly["geo"].unique()))
    print(quarterly.groupby("geo").tail(3)[
        ["geo", "period", "hicp_index", "inflation_qoq_pct", "inflation_qoq_pct_l1"]
    ].to_string(index=False))

if __name__ == "__main__":
    main()