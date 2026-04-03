from pathlib import Path
import pandas as pd

# Raíz del repo: subimos desde src/app/web/backend/ → 4 niveles
REPO_ROOT   = Path(__file__).resolve().parents[4]
REPORTS_DIR = REPO_ROOT / "reports"

GEO_LABELS = {"ES": "España", "FR": "Francia", "IT": "Italia"}

# Nombres de CSV por bloque
def metrics_path(bloque: int, geo: str) -> Path:
    geo_lower = geo.lower()
    if bloque == 0:
        return REPORTS_DIR / f"metrics_xgboost_{geo_lower}_compare.csv"
    elif bloque == 1:
        return REPORTS_DIR / "metrics_panel_combined_compare.csv"
    elif bloque == 2:
        return REPORTS_DIR / "metrics_panel_geo_compare.csv"
    elif bloque == 3:
        return REPORTS_DIR / "metrics_panel_transfer_compare.csv"
    raise ValueError(f"Bloque desconocido: {bloque}")


def predictions_path(bloque: int, geo: str, modelo: str) -> Path:
    geo_lower = geo.lower()
    if bloque == 0:
        return REPORTS_DIR / f"predictions_xgboost_{modelo}_{geo_lower}.csv"
    elif bloque == 1:
        return REPORTS_DIR / f"predictions_panel_combined_{modelo}_{geo_lower}.csv"
    elif bloque == 2:
        return REPORTS_DIR / f"predictions_panel_geo_{modelo}_{geo_lower}.csv"
    elif bloque == 3:
        # Solo existe modelo ext
        return REPORTS_DIR / f"predictions_panel_transfer_ext_{geo_lower}.csv"
    raise ValueError(f"Bloque desconocido: {bloque}")


def load_metrics(bloque: int, geo: str) -> pd.DataFrame:
    path = metrics_path(bloque, geo)
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    # Bloques 1, 2, 3 tienen columna geo → filtrar
    if bloque in (1, 2, 3) and "geo" in df.columns:
        df = df[df["geo"] == geo].reset_index(drop=True)
    return df


def load_predictions(bloque: int, geo: str, modelo: str) -> pd.DataFrame:
    path = predictions_path(bloque, geo, modelo)
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    if "period" in df.columns:
        df["period"] = df["period"].astype(str)
    return df


def load_all_metrics() -> pd.DataFrame:
    """Carga y combina métricas de todos los bloques para la tabla resumen global."""
    frames = []

    # Bloque 0: un CSV por país
    for geo in GEO_LABELS:
        p = REPORTS_DIR / f"metrics_xgboost_{geo.lower()}_compare.csv"
        if p.exists():
            df = pd.read_csv(p)
            df["bloque"] = 0
            df["geo"] = geo
            frames.append(df)

    # Bloques 1, 2, 3: CSV único con columna geo
    for bloque, fname in [
        (1, "metrics_panel_combined_compare.csv"),
        (2, "metrics_panel_geo_compare.csv"),
        (3, "metrics_panel_transfer_compare.csv"),
    ]:
        p = REPORTS_DIR / fname
        if p.exists():
            df = pd.read_csv(p)
            df["bloque"] = bloque
            frames.append(df)

    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)
