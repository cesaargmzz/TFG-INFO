from pathlib import Path
import pandas as pd

RAW_PATH = Path("../../data/raw/gdp_spain_api.parquet")
OUT_PATH = Path("../../data/processed/gdp_spain_qoq.parquet")

df = pd.read_parquet(RAW_PATH).sort_values("period").reset_index(drop=True)

# Asegurar tipo Period trimestral
df["period"] = pd.PeriodIndex(df["period"], freq="Q")

# Calcular variación trimestral %
df["gdp_qoq_pct"] = df["value"].astype("float64").pct_change() * 100

# Renombrar nivel para claridad
df = df.rename(columns={"value": "gdp_level"})

# Eliminar primera fila NaN
df = df.dropna().reset_index(drop=True)

OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
df.to_parquet(OUT_PATH, index=False)

print("Guardado en:", OUT_PATH.resolve())
print(df.tail(8))
