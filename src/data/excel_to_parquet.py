from pathlib import Path
import pandas as pd

# Rutas
RAW_XLSX = Path("../../data/raw/gdp_spain.xlsx")
OUT_PARQUET = Path("../../data/processed/gdp_spain.parquet")

def _to_float_eu(x):
    if pd.isna(x):
        return pd.NA
    s = str(x).strip()
    if s == "":
        return pd.NA

    try:
        return float(s)
    except ValueError:
        return pd.NA

def main():
    if not RAW_XLSX.exists():
        raise FileNotFoundError(f"No encuentro el archivo: {RAW_XLSX.resolve()}")

    df = pd.read_excel(RAW_XLSX)

    if df.shape[1] < 3:
        raise ValueError("El excel no tiene el formato esperado")
    
    geo_col = df.columns[0]
    time_label_col = df.columns[1]

    long_df = df.melt(
        id_vars=[geo_col, time_label_col],
        var_name="time",
        value_name="value_raw"
    )

    long_df = long_df.rename(columns={geo_col: "geo"})

    long_df["time"] = long_df["time"].astype(str).str.strip()
    long_df = long_df[long_df["time"].str.match(r"^\d{4}-Q[1-4]$", na=False)].copy()

    long_df["period"] = pd.PeriodIndex(long_df["time"], freq="Q")

    long_df["value"] = (
        long_df["value_raw"]
        .astype(str)
        .str.replace(",", ".", regex=False)
    )

    long_df["value"] = pd.to_numeric(long_df["value"], errors="coerce")

    long_df = long_df.dropna(subset=["value"]).copy()

    long_df = long_df[["geo", "period", "value"]].sort_values(["geo", "period"]).reset_index(drop=True)

    OUT_PARQUET.parent.mkdir(parents=True, exist_ok=True)
    long_df.to_parquet(OUT_PARQUET, index=False)

    print("✅ Parquet creado:", OUT_PARQUET.as_posix())
    print("Filas:", len(long_df), "| Geos:", long_df["geo"].nunique(),
          "| Desde:", long_df["period"].min(), "| Hasta:", long_df["period"].max())

if __name__ == "__main__":
    main()