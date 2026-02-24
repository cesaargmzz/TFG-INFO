from pathlib import Path
import pandas as pd

GDP_PATH = Path("../../data/processed/gdp_qoq.parquet")
UNEMP_PATH = Path("../../data/raw/unemployment_api.parquet")
INFL_PATH = Path("../../data/processed/inflation_qoq.parquet")

# Nuevas variables (trimestrales QoQ)
IPI_PATH = Path("../../data/processed/ipi_qoq.parquet")
RETAIL_PATH = Path("../../data/processed/retail_qoq.parquet")

OUT_PATH = Path("../../data/processed/dataset_v3.parquet")

gdp = pd.read_parquet(GDP_PATH).copy()
unemp = pd.read_parquet(UNEMP_PATH).copy()
infl = pd.read_parquet(INFL_PATH).copy()
ipi = pd.read_parquet(IPI_PATH).copy()
retail = pd.read_parquet(RETAIL_PATH).copy()

# Asegurar claves
for df in (gdp, unemp, infl, ipi, retail):
    if "geo" not in df.columns:
        df["geo"] = "ES"

# (Opcional) Asegurar que 'period' sea trimestral si viniera como string
# Si ya viene como Period[Q], no pasa nada.
for df in (gdp, unemp, infl, ipi, retail):
    if df["period"].dtype == object:
        # intenta parsear tipo '2009Q2'
        df["period"] = pd.PeriodIndex(df["period"], freq="Q")

# Merge por geo + period (inner = intersección de periodos comunes)
df = gdp.merge(unemp, on=["geo", "period"], how="inner")
df = df.merge(infl[["geo", "period", "inflation_qoq_pct"]], on=["geo", "period"], how="inner")

# Nuevas variables
df = df.merge(ipi[["geo", "period", "ipi_qoq_pct"]], on=["geo", "period"], how="inner")
df = df.merge(retail[["geo", "period", "retail_qoq_pct"]], on=["geo", "period"], how="inner")

df = df.sort_values(["geo", "period"]).reset_index(drop=True)

# Lags del PIB (para que estén ya en el dataset y sea reutilizable)
df["gdp_qoq_pct_l1"] = df.groupby("geo")["gdp_qoq_pct"].shift(1)
df["gdp_qoq_pct_l2"] = df.groupby("geo")["gdp_qoq_pct"].shift(2)

# Lags explicativas
df["unemployment_rate_l1"] = df.groupby("geo")["unemployment_rate"].shift(1)
df["inflation_qoq_pct_l1"] = df.groupby("geo")["inflation_qoq_pct"].shift(1)
df["ipi_qoq_pct_l1"] = df.groupby("geo")["ipi_qoq_pct"].shift(1)
df["retail_qoq_pct_l1"] = df.groupby("geo")["retail_qoq_pct"].shift(1)

print("Shape:", df.shape)
print(df.tail(10).to_string(index=False))

OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
df.to_parquet(OUT_PATH, index=False)
print("Guardado en:", OUT_PATH.resolve())
