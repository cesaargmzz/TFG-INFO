from pathlib import Path
import pandas as pd

PARQUET_PATH = Path("../../data/processed/gdp_spain.parquet")
OUT_PARQUET = Path("../../data/processed/gdp_spain_qoq.parquet")

df = pd.read_parquet(PARQUET_PATH).sort_values("period").reset_index(drop=True)

# Variación trimestral (%)
df["value_qoq_pct"] = df["value"].astype("float64").pct_change() * 100

# (opcional) quitar la primera fila NaN si te molesta
df_qoq = df.dropna(subset=["value_qoq_pct"]).copy()

# Guardar nuevo parquet con la serie transformada
OUT_PARQUET.parent.mkdir(parents=True, exist_ok=True)
df_qoq.to_parquet(OUT_PARQUET, index=False)

print("Guardado:", OUT_PARQUET.as_posix())
print(df_qoq[["period", "value", "value_qoq_pct"]].tail(12).to_string(index=False))
