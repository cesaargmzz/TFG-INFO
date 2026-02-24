import requests
import pandas as pd
from pathlib import Path

# Dataset PIB trimestral
DATASET = "namq_10_gdp"

out_path = Path("../../data/raw/gdp_spain_api.parquet")
out_path.parent.mkdir(parents=True, exist_ok=True)

url = (
    f"https://ec.europa.eu/eurostat/api/dissemination/statistics/1.0/data/"
    f"{DATASET}"
    "?geo=ES"
    "&na_item=B1GQ"
    "&unit=CLV10_MEUR"
    "&s_adj=SCA"
)

response = requests.get(url)
data = response.json()

# Extraer valores
values = data["value"]
time_index = data["dimension"]["time"]["category"]["index"]

records = []

for period, idx in time_index.items():
    if str(idx) in values:
        records.append({
            "period": period,
            "value": values[str(idx)]
        })

df = pd.DataFrame(records).sort_values("period")
df["period"] = pd.PeriodIndex(df["period"], freq="Q")

df.to_parquet(out_path, index=False)

print("Guardado en:", out_path.resolve())

print(df.tail(12))
