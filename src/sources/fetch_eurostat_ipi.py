from pathlib import Path
import requests
import pandas as pd

# Eurostat dataset: Production in industry - monthly data
# Online data code: sts_inpr_m
DATASET = "sts_inpr_m"

# Parámetros típicos:
# - indic_bt=PRD (production)
# - s_adj=SCA (seasonally and calendar adjusted)
# - unit=I21 (Index, 2021=100)
# - nace_r2=B-D (Industry excluding construction)  -> si falla, cambia a "B-E" o prueba sin nace_r2
#
# OJO: Eurostat a veces cambia códigos disponibles por dimensión.
# Si te da 400, prueba:
#  - quitar nace_r2
#  - cambiar s_adj (ej. "SA", "NSA")
#  - cambiar unit (ej. "I15", etc.)

GEO = "ES"
PARAMS = {
    "geo": GEO,
    "indic_bt": "PRD",
    "s_adj": "SCA",
    "unit": "I21",
    "nace_r2": "B-D",
}

BASE_URL = "https://ec.europa.eu/eurostat/api/dissemination/statistics/1.0/data"

OUT_PATH = Path("../../data/raw/ipi_es_api.parquet")


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
            records.append({"period": period, "ipi_index": float(v)})

    df = pd.DataFrame(records).sort_values("period").reset_index(drop=True)
    df["period"] = pd.PeriodIndex(df["period"], freq="M")
    df["geo"] = GEO

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT_PATH, index=False)

    print("Guardado en:", OUT_PATH.resolve())
    print(df.tail(12).to_string(index=False))


if __name__ == "__main__":
    main()
