from pathlib import Path
import pandas as pd

DATA_DIR = Path(__file__).resolve().parents[2] / "data"


def load_pib(path: Path | None = None) -> pd.DataFrame:
    """
    Carga el PIB trimestral desde CSV y devuelve un DataFrame normalizado:
    - Index: PeriodIndex trimestral
    - Columna: 'pib_var' (variación del PIB, tal como venga en el CSV)
    """
    if path is None:
        path = DATA_DIR / "PIB.csv"
    df = pd.read_csv(path, sep=";", encoding="utf-8")[["PERIODO", "VALOR"]].copy()
    df.columns = ["Trimestre", "pib_var"]
    df["Trimestre"] = df["Trimestre"].str.replace("T", "Q", regex=False)
    df["Trimestre"] = pd.PeriodIndex(df["Trimestre"], freq="Q")
    df = df.set_index("Trimestre").sort_index()
    return df


def save_parquet(df: pd.DataFrame, name: str, subdir: str = "processed"):
    out = DATA_DIR / subdir / f"{name}.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=True)
    return out
