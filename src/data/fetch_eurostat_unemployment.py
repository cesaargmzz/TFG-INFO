from pathlib import Path
import requests
import pandas as pd

DATASET = "une_rt_q"
GEO = "ES"  # cambia aquí el país

url = (
    "https://ec.europa.eu/eurostat/api/dissemination/statistics/1.0/data/"
    f"{DATASET}"
    f"?geo={GEO}"
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
        records.append({
            "geo": GEO,
            "period": period,
            "unemployment_rate": float(values[key])
        })

df_new = pd.DataFrame(records).sort_values("period").reset_index(drop=True)
df_new["period"] = pd.PeriodIndex(df_new["period"], freq="Q")

out_path = Path("../../data/raw/unemployment_api.parquet")
out_path.parent.mkdir(parents=True, exist_ok=True)

# Si ya existe, concatenamos
if out_path.exists():
    df_existing = pd.read_parquet(out_path)
    df = pd.concat([df_existing, df_new], ignore_index=True)
    df = df.drop_duplicates(subset=["geo", "period"]).sort_values(["geo", "period"])
else:
    df = df_new

df.to_parquet(out_path, index=False)

print("Guardado en:", out_path.resolve())
print(df.tail(12).to_string(index=False))
