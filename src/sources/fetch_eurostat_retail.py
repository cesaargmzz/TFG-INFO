from pathlib import Path
import requests
import pandas as pd

# Eurostat dataset: Turnover and volume of sales in wholesale and retail trade - monthly data
# Online data code: sts_trtu_m
DATASET = "sts_trtu_m"

# Objetivo: serie de "retail trade volume index" (volumen) para ES.
# Dimensiones típicas incluyen:
# - s_adj (ajuste estacional)
# - unit (índice base)
# - nace_r2 (actividad)
# - indic_bt o similar (según dataset)
#
# Si te falla (400), empieza quitando nace_r2 o ajustando unit/s_adj.

GEO = "ES"
PARAMS = {
    "geo": GEO,
    "s_adj": "SCA",
    "unit": "I21",
    # Total retail trade (aprox.). Si falla, prueba con "G47" o elimina este filtro.
    "nace_r2": "G47",
}

BASE_URL = "https://ec.europa.eu/eurostat/api/dissemination/statistics/1.0/data"

OUT_PATH = Path("../../data/raw/retail_es_api.parquet")


def main():
    url = f"{BASE_URL}/{DATASET}"
    resp = requests.get(url, params=PARAMS, timeout=60)
    resp.raise_for_status()
    data = resp.json()

    values = data.get("value", {})
    time_index = data["dimension"]["time"]["category"]["index"]

    records = []
    for period, idx in time_index.items():
        key = str(idx)
        v = values.get(key, None)
        if v is not None:
            records.append({"period": period, "retail_index": float(v)})

    df = pd.DataFrame(records).sort_values("period").reset_index(drop=True)
    df["period"] = pd.PeriodIndex(df["period"], freq="M")
    df["geo"] = GEO

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT_PATH, index=False)

    print("Guardado en:", OUT_PATH.resolve())
    print(df.tail(12).to_string(index=False))


if __name__ == "__main__":
    main()
