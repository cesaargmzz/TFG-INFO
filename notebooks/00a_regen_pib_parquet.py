# notebooks/00a_regen_pib_parquet.py

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from src.data.loaders import load_pib, save_parquet


def main():
    df = load_pib()  # usa data/PIB.csv
    print("\nDtypes tras la carga:")
    print(df.dtypes)
    print("\nPrimeras filas:")
    print(df.head())

    out = save_parquet(df, "pib_trimestral")
    print(f"\nParquet regenerado en: {out}")


if __name__ == "__main__":
    main()
