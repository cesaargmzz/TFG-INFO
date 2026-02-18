from pathlib import Path
import pandas as pd

GDP_PATH = Path("../../data/processed/gdp_qoq.parquet")
UNEMP_PATH = Path("../../data/raw/unemployment_api.parquet")
INFL_PATH = Path("../../data/processed/inflation_qoq.parquet")

OUT_PATH = Path("../../data/processed/dataset_v1.parquet")

gdp = pd.read_parquet(GDP_PATH).copy()
unemp = pd.read_parquet(UNEMP_PATH).copy()
infl = pd.read_parquet(INFL_PATH).copy()

# Asegurar claves
for df in (gdp, unemp, infl):
    if "geo" not in df.columns:
        df["geo"] = "ES"

# Merge por geo + period
df = gdp.merge(unemp, on=["geo", "period"], how="inner")
df = df.merge(infl[["geo", "period", "inflation_qoq_pct"]], on=["geo", "period"], how="inner")

df = df.sort_values(["geo", "period"]).reset_index(drop=True)

print("Shape:", df.shape)
print(df.tail(10).to_string(index=False))

OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
df.to_parquet(OUT_PATH, index=False)
print("Guardado en:", OUT_PATH.resolve())
