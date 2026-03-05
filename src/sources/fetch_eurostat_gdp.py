import requests
import pandas as pd
from pathlib import Path
import argparse

# Dataset PIB trimestral
DATASET = "namq_10_gdp"

def fetch_gdp_for_geo(geo: str) -> pd.DataFrame:
    url = (
        f"https://ec.europa.eu/eurostat/api/dissemination/statistics/1.0/data/"
        f"{DATASET}"
        f"?geo={geo}"
        f"&na_item=B1GQ"
        f"&unit=CLV10_MEUR"
        f"&s_adj=SCA"
    )

    response = requests.get(url, timeout=60)
    response.raise_for_status()
    data = response.json()

    values = data.get("value", {})
    time_index = data["dimension"]["time"]["category"]["index"]

    records = []
    for period, idx in time_index.items():
        if str(idx) in values:
            records.append({
                "geo": geo,
                "period": period,
                "value": values[str(idx)]
            })

    df = pd.DataFrame(records).sort_values("period")
    df["period"] = pd.PeriodIndex(df["period"], freq="Q")
    return df

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--geos", nargs="+", default=["ES", "IT", "FR"], help="Lista de países (geo), ej: ES IT FR")
    ap.add_argument("--out", default="../../data/raw/gdp_eurostat_api.parquet", help="Ruta de salida parquet")
    args = ap.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    frames = []
    for geo in args.geos:
        frames.append(fetch_gdp_for_geo(geo))

    df = pd.concat(frames, ignore_index=True)
    df = df.sort_values(["geo", "period"])

    df.to_parquet(out_path, index=False)

    print("Guardado en:", out_path.resolve())
    print("Geos:", sorted(df["geo"].unique()))
    print(df.groupby("geo").tail(3))

if __name__ == "__main__":
    main()