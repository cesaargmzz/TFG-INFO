from pathlib import Path
import argparse
import requests
import pandas as pd

DATASET = "sts_inpr_m"

BASE_URL = "https://ec.europa.eu/eurostat/api/dissemination/statistics/1.0/data"


def fetch_ipi_for_geo(geo: str):

    params = {
        "geo": geo,
        "indic_bt": "PRD",
        "s_adj": "SCA",
        "unit": "I21",
        "nace_r2": "B-D",
    }

    url = f"{BASE_URL}/{DATASET}"

    resp = requests.get(url, params=params, timeout=60)
    resp.raise_for_status()
    data = resp.json()

    values = data.get("value", {})
    time_index = data["dimension"]["time"]["category"]["index"]

    records = []

    for period, idx in time_index.items():
        key = str(idx)
        v = values.get(key)

        if v is not None:
            records.append({
                "geo": geo,
                "period": period,
                "ipi_index": float(v)
            })

    df = pd.DataFrame(records).sort_values("period").reset_index(drop=True)

    df["period"] = pd.PeriodIndex(df["period"], freq="M")

    return df


def main():

    ap = argparse.ArgumentParser()
    ap.add_argument("--geos", nargs="+", default=["ES", "IT", "FR"])
    ap.add_argument("--out", default="../../data/raw/ipi_api.parquet")

    args = ap.parse_args()

    frames = [fetch_ipi_for_geo(g) for g in args.geos]

    df = pd.concat(frames, ignore_index=True)

    df = df.sort_values(["geo", "period"])

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    df.to_parquet(out_path, index=False)

    print("Guardado en:", out_path.resolve())
    print("Geos:", sorted(df["geo"].unique()))
    print(df.groupby("geo").tail(6))


if __name__ == "__main__":
    main()