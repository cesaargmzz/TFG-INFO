from pathlib import Path
import pandas as pd

RAW_PATH = Path("../../data/raw/inflation_es_api.parquet")
OUT_PATH = Path("../../data/processed/inflation_qoq.parquet")

df = pd.read_parquet(RAW_PATH).sort_values(["period"]).reset_index(drop=True)

# Asegurar Period mensual
df["period"] = pd.PeriodIndex(df["period"], freq="M")

# Añadir geo si no existe (por ahora ES)
if "geo" not in df.columns:
    df["geo"] = "ES"

# Trimestre
df["quarter"] = df["period"].dt.to_timestamp().dt.to_period("Q")

# Media trimestral del índice
q = (
    df.groupby(["geo", "quarter"])["hicp_index"]
      .mean()
      .reset_index()
      .rename(columns={"quarter": "period", "hicp_index": "hicp_q_avg"})
      .sort_values(["geo", "period"])
      .reset_index(drop=True)
)

# Variación trimestral (%) del índice trimestral
q["inflation_qoq_pct"] = q.groupby("geo")["hicp_q_avg"].pct_change() * 100

# Limpiar
q = q.dropna(subset=["inflation_qoq_pct"]).reset_index(drop=True)

OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
q.to_parquet(OUT_PATH, index=False)

print("Guardado en:", OUT_PATH.resolve())
print(q.tail(8).to_string(index=False))
