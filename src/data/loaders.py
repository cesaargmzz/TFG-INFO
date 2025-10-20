# src/data/loaders.py
from pathlib import Path
import pandas as pd

DATA_DIR = Path(__file__).resolve().parents[2] / "data"


def _to_float_from_es_decimal(s: pd.Series) -> pd.Series:
    """
    Convierte series con coma decimal (p. ej. '0,6592') a float.
    También elimina posibles espacios en blanco.
    Devuelve una serie float con NaN si algo no se puede convertir.
    """
    s = s.astype(str).str.strip()
    s = s.str.replace(",", ".", regex=False)  # coma -> punto decimal
    return pd.to_numeric(s, errors="coerce")


def load_pib(path: Path | None = None) -> pd.DataFrame:
    """
    Carga el PIB trimestral desde CSV y devuelve un DataFrame normalizado:
    - Index: PeriodIndex trimestral
    - Columna: 'pib_var' (float, variación del PIB)
    """
    if path is None:
        path = DATA_DIR / "PIB.csv"

    # Lee CSV del INE (suele venir con ; como separador)
    df = pd.read_csv(path, sep=";", encoding="utf-8")[["PERIODO", "VALOR"]].copy()
    df.columns = ["Trimestre", "pib_var"]

    # Trimestre como PeriodIndex
    df["Trimestre"] = df["Trimestre"].str.replace("T", "Q", regex=False)
    df["Trimestre"] = pd.PeriodIndex(df["Trimestre"], freq="Q")

    # ---- CONVERSIÓN A FLOAT (arreglo clave) ----
    df["pib_var"] = _to_float_from_es_decimal(df["pib_var"])

    # Índice y orden
    df = df.set_index("Trimestre").sort_index()
    return df


def save_parquet(df: pd.DataFrame, name: str, subdir: str = "processed"):
    out = DATA_DIR / subdir / f"{name}.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=True)
    return out
