from pathlib import Path
import argparse
import requests
import pandas as pd

DATASET = "une_rt_q"

def fetch_unemp_for_geo(geo: str) -> pd.DataFrame:
    url = (
        "https://ec.europa.eu/eurostat/api/dissemination/statistics/1.0/data/"
        f"{DATASET}"
        f"?geo={geo}"
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
                "geo": geo,
                "period": period,
                "unemployment_rate": float(values[key])
            })

    df = pd.DataFrame(records).sort_values("period").reset_index(drop=True)
    df["period"] = pd.PeriodIndex(df["period"], freq="Q")
    return df

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--geos", nargs="+", default=["ES", "IT", "FR"])
    ap.add_argument("--out", default="../../data/raw/unemployment_api.parquet")
    args = ap.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    frames = [fetch_unemp_for_geo(g) for g in args.geos]
    df = pd.concat(frames, ignore_index=True).sort_values(["geo", "period"])

    df.to_parquet(out_path, index=False)

    print("Guardado en:", out_path.resolve())
    print("Geos:", sorted(df["geo"].unique()))
    print(df.groupby("geo").tail(3).to_string(index=False))

if __name__ == "__main__":
    main()