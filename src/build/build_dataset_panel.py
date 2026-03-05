import argparse
from pathlib import Path
import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gdp", default="../../data/processed/gdp_qoq_panel.parquet")
    ap.add_argument("--unemp", default="../../data/processed/unemployment_panel.parquet")
    ap.add_argument("--infl", default="../../data/processed/inflation_qoq_panel.parquet")
    ap.add_argument("--ipi", default="../../data/processed/ipi_qoq_panel.parquet")
    ap.add_argument("--retail", default="../../data/processed/retail_qoq_panel.parquet")
    ap.add_argument("--out", default="../../data/processed/dataset_panel_v1.parquet")
    args = ap.parse_args()

    gdp = pd.read_parquet(Path(args.gdp)).copy()
    unemp = pd.read_parquet(Path(args.unemp)).copy()
    infl = pd.read_parquet(Path(args.infl)).copy()
    ipi = pd.read_parquet(Path(args.ipi)).copy()
    retail = pd.read_parquet(Path(args.retail)).copy()

    # --- Seleccionamos columnas necesarias y renombramos medias trimestrales si aparecen ---
    gdp = gdp[["geo", "period", "gdp_level", "gdp_qoq_pct", "gdp_qoq_pct_l1", "gdp_qoq_pct_l2"]]

    unemp = unemp[["geo", "period", "unemployment_rate", "unemployment_rate_l1"]]

    infl = infl[["geo", "period", "hicp_index", "inflation_qoq_pct", "inflation_qoq_pct_l1"]]

    # ipi transform deja ipi_q_avg / ipi_qoq_pct / ipi_qoq_pct_l1
    ipi = ipi[["geo", "period", "ipi_q_avg", "ipi_qoq_pct", "ipi_qoq_pct_l1"]]

    # retail transform deja retail_q_avg / retail_qoq_pct / retail_qoq_pct_l1
    retail = retail[["geo", "period", "retail_q_avg", "retail_qoq_pct", "retail_qoq_pct_l1"]]

    # --- Merge por (geo, period) ---
    df = gdp.merge(unemp, on=["geo", "period"], how="left") \
            .merge(infl, on=["geo", "period"], how="left") \
            .merge(ipi, on=["geo", "period"], how="left") \
            .merge(retail, on=["geo", "period"], how="left")

    df = df.sort_values(["geo", "period"]).reset_index(drop=True)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)

    print("Guardado en:", out_path.resolve())
    print("Geos:", sorted(df["geo"].unique()))
    print("Rango periodos:", df["period"].min(), "->", df["period"].max())
    print("Filas por geo:")
    print(df.groupby("geo").size())

    print("\nColumnas:")
    print(list(df.columns))

    # sanity: últimas 2 filas por país
    print("\nTail por geo:")
    print(df.groupby("geo").tail(2)[["geo","period","gdp_qoq_pct","unemployment_rate_l1","inflation_qoq_pct_l1","ipi_qoq_pct_l1","retail_qoq_pct_l1"]].to_string(index=False))


if __name__ == "__main__":
    main()