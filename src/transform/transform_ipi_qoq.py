from pathlib import Path
import pandas as pd

RAW_PATH = Path("../../data/raw/ipi_api.parquet")
OUT_PATH = Path("../../data/processed/ipi_qoq_panel.parquet")


def main():

    df = pd.read_parquet(RAW_PATH).sort_values(["geo", "period"]).reset_index(drop=True)

    # asegurar period mensual
    df["period"] = pd.PeriodIndex(df["period"], freq="M")

    # convertir a trimestre
    df["quarter"] = df["period"].dt.to_timestamp().dt.to_period("Q")

    # media trimestral
    q = (
        df.groupby(["geo", "quarter"])["ipi_index"]
        .mean()
        .reset_index()
        .rename(columns={
            "quarter": "period",
            "ipi_index": "ipi_q_avg"
        })
        .sort_values(["geo", "period"])
        .reset_index(drop=True)
    )

    # variación trimestral %
    q["ipi_qoq_pct"] = q.groupby("geo")["ipi_q_avg"].pct_change() * 100

    # lag
    q["ipi_qoq_pct_l1"] = q.groupby("geo")["ipi_qoq_pct"].shift(1)

    q = q.dropna(subset=["ipi_qoq_pct"]).reset_index(drop=True)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    q.to_parquet(OUT_PATH, index=False)

    print("Guardado en:", OUT_PATH.resolve())
    print("Geos:", sorted(q["geo"].unique()))
    print(q.groupby("geo").tail(3).to_string(index=False))


if __name__ == "__main__":
    main()