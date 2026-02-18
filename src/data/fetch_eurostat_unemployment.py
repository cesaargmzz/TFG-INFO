from pathlib import Path
import requests
import pandas as pd

DATASET = "une_rt_q"

url = (
    "https://ec.europa.eu/eurostat/api/dissemination/statistics/1.0/data/"
    f"{DATASET}"
    "?geo=ES"
    "&s_adj=SA"
    "&unit=PC_ACT"
    "&sex=T"
    "&age=Y15-74"
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
        records.append({"period": period, "unemployment_rate": float(values[key])})

df = pd.DataFrame(records).sort_values("period").reset_index(drop=True)
df["period"] = pd.PeriodIndex(df["period"], freq="Q")

# Guardar en raw
out_path = Path("../../data/raw/unemployment_es_api.parquet")
out_path.parent.mkdir(parents=True, exist_ok=True)
df.to_parquet(out_path, index=False)

print("Guardado en:", out_path.resolve())
print(df.tail(12).to_string(index=False))
