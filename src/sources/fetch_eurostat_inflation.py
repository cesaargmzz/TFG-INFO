from pathlib import Path
import requests
import pandas as pd

DATASET = "prc_hicp_midx"

url = (
    "https://ec.europa.eu/eurostat/api/dissemination/statistics/1.0/data/"
    f"{DATASET}"
    "?geo=ES"
    "&coicop=CP00"
    "&unit=I15"
)

resp = requests.get(url, timeout=60)
resp.raise_for_status()
data = resp.json()

values = data.get("value", {})
time_index = data["dimension"]["time"]["category"]["index"]

records = []
for period, idx in time_index.items():
    key = str(idx)
    if key in values and values[key] is not None:
        records.append({
            "period": period,
            "hicp_index": float(values[key])
        })

df = pd.DataFrame(records).sort_values("period").reset_index(drop=True)

# Period mensual
df["period"] = pd.PeriodIndex(df["period"], freq="M")

out_path = Path("../../data/raw/inflation_es_api.parquet")
out_path.parent.mkdir(parents=True, exist_ok=True)
df.to_parquet(out_path, index=False)

print("Guardado en:", out_path.resolve())
print(df.tail(12))
