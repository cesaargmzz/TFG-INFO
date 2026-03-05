from pathlib import Path
import pandas as pd

RAW_PATH = Path("../../data/raw/retail_api.parquet")
OUT_PATH = Path("../../data/processed/retail_qoq_panel.parquet")


def main():

    df = pd.read_parquet(RAW_PATH).sort_values(["geo", "period"]).reset_index(drop=True)

    # asegurar period mensual
    df["period"] = pd.PeriodIndex(df["period"], freq="M")

    # convertir a trimestre
    df["quarter"] = df["period"].dt.to_timestamp().dt.to_period("Q")

    # media trimestral
    q = (
        df.groupby(["geo", "quarter"])["retail_index"]
        .mean()
        .reset_index()
        .rename(columns={
            "quarter": "period",
            "retail_index": "retail_q_avg"
        })
        .sort_values(["geo", "period"])
        .reset_index(drop=True)
    )

    # variación trimestral %
    q["retail_qoq_pct"] = q.groupby("geo")["retail_q_avg"].pct_change() * 100

    # lag
    q["retail_qoq_pct_l1"] = q.groupby("geo")["retail_qoq_pct"].shift(1)

    q = q.dropna(subset=["retail_qoq_pct"]).reset_index(drop=True)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    q.to_parquet(OUT_PATH, index=False)

    print("Guardado en:", OUT_PATH.resolve())
    print("Geos:", sorted(q["geo"].unique()))
    print(q.groupby("geo").tail(3).to_string(index=False))


if __name__ == "__main__":
    main()