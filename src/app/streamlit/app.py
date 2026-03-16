import pandas as pd
import plotly.express as px
import streamlit as st
from pathlib import Path

# --- Paths (repo root assumed) ---
REPO_ROOT = Path(__file__).resolve().parents[3]  # src/app/streamlit/app.py -> repo root
REPORTS_DIR = REPO_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

GEO_LABELS = {"ES": "España", "FR": "Francia", "IT": "Italia"}

BLOQUE_OPTIONS = {
    "Bloque 0 — Por país": 0,
    "Bloque 1 — Panel combinado": 1,
    "Bloque 2 — Panel con geo como feature": 2,
}

st.set_page_config(page_title="TFG Dashboard — PIB QoQ", layout="wide")

# --- Helpers ---
def load_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        st.error(f"No se encontró el archivo: {path}")
        st.stop()
    return pd.read_csv(path)

def safe_period_str(df: pd.DataFrame) -> pd.DataFrame:
    if "period" in df.columns:
        df["period"] = df["period"].astype(str)
    return df

# --- Sidebar controls ---
st.sidebar.header("Controles")

bloque_label = st.sidebar.selectbox("Bloque experimental", list(BLOQUE_OPTIONS.keys()), index=0)
bloque = BLOQUE_OPTIONS[bloque_label]

geo = st.sidebar.selectbox("País", list(GEO_LABELS.keys()),
                            format_func=lambda g: f"{g} — {GEO_LABELS[g]}", index=0)

if bloque == 0:
    model_options = {
        "XGBoost (base)": "base",
        "XGBoost (ampliado)": "ext",
    }
elif bloque == 1:
    model_options = {
        "Panel combinado (base)": "base",
        "Panel combinado (ampliado)": "ext",
    }
else:
    model_options = {
        "Panel geo (base)": "base",
        "Panel geo (ampliado)": "ext",
    }
model_label = st.sidebar.selectbox("Modelo", list(model_options.keys()), index=0)
model_key = model_options[model_label]

show_tables = st.sidebar.checkbox("Mostrar tablas", value=True)
show_figures_folder = st.sidebar.checkbox("Mostrar carpeta de figuras (PNG)", value=False)

# --- Dynamic paths ---
geo_lower = geo.lower()

if bloque == 0:
    metrics_path   = REPORTS_DIR / f"metrics_xgboost_{geo_lower}_compare.csv"
    pred_base_path = REPORTS_DIR / f"predictions_xgboost_base_{geo_lower}.csv"
    pred_ext_path  = REPORTS_DIR / f"predictions_xgboost_ext_{geo_lower}.csv"
    fig_subdir     = FIGURES_DIR / "bloque0"
elif bloque == 1:
    metrics_path   = REPORTS_DIR / "metrics_panel_combined_compare.csv"
    pred_base_path = REPORTS_DIR / f"predictions_panel_combined_base_{geo_lower}.csv"
    pred_ext_path  = REPORTS_DIR / f"predictions_panel_combined_ext_{geo_lower}.csv"
    fig_subdir     = FIGURES_DIR / "bloque1"
else:
    metrics_path   = REPORTS_DIR / "metrics_panel_geo_compare.csv"
    pred_base_path = REPORTS_DIR / f"predictions_panel_geo_base_{geo_lower}.csv"
    pred_ext_path  = REPORTS_DIR / f"predictions_panel_geo_ext_{geo_lower}.csv"
    fig_subdir     = FIGURES_DIR / "bloque2"

# --- Load data ---
metrics_raw = load_csv(metrics_path)
if bloque in (1, 2) and "geo" in metrics_raw.columns:
    metrics = metrics_raw[metrics_raw["geo"] == geo].reset_index(drop=True)
else:
    metrics = metrics_raw

pred_base = safe_period_str(load_csv(pred_base_path))
pred_ext  = safe_period_str(load_csv(pred_ext_path))

# --- Title (dynamic) ---
st.title(f"Dashboard — Predicción del PIB QoQ · {geo} ({GEO_LABELS[geo]}) · {bloque_label}")
st.caption("Visualización interactiva de resultados de modelos (CSV generados por el pipeline).")

# --- Select predictions ---
if model_key == "base":
    preds = pred_base.copy()
    metrics_row = metrics[metrics["model"].str.contains("base", case=False, na=False)]
else:
    preds = pred_ext.copy()
    metrics_row = metrics[metrics["model"].str.contains("ext", case=False, na=False)]

if metrics_row.empty:
    metrics_row = metrics.head(1)

mae_val      = float(metrics_row["mae"].iloc[0])      if "mae"        in metrics_row.columns else None
rmse_val     = float(metrics_row["rmse"].iloc[0])     if "rmse"       in metrics_row.columns else None
n_features   = int(metrics_row["n_features"].iloc[0]) if "n_features" in metrics_row.columns else None

# --- KPIs ---
kpi1, kpi2, kpi3 = st.columns(3)
kpi1.metric("Modelo", model_label)
kpi2.metric("MAE (p.p.)", f"{mae_val:.3f}"  if mae_val  is not None else "—")
kpi3.metric("RMSE (p.p.)", f"{rmse_val:.3f}" if rmse_val is not None else "—")

if n_features is not None:
    st.caption(f"Nº de variables: {n_features}")

st.divider()

# --- Plots ---
col1, col2 = st.columns(2)

with col1:
    st.subheader("Real vs Predicción (test)")
    if {"period", "y_true", "y_pred"}.issubset(preds.columns):
        fig = px.line(
            preds,
            x="period",
            y=["y_true", "y_pred"],
            markers=True,
            labels={"value": "PIB QoQ (%)", "period": "Trimestre", "variable": ""},
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("El CSV no contiene las columnas esperadas: period, y_true, y_pred")

with col2:
    st.subheader("Errores (Real - Pred)")
    if {"period", "error"}.issubset(preds.columns):
        fig2 = px.line(
            preds,
            x="period",
            y="error",
            markers=True,
            labels={"error": "Error (p.p.)", "period": "Trimestre"},
        )
        fig2.add_hline(y=0, line_dash="dash")
        st.plotly_chart(fig2, use_container_width=True)
    else:
        st.warning("El CSV no contiene las columnas esperadas: period, error")

st.divider()

# --- Tables ---
if show_tables:
    st.subheader("Predicciones (conjunto de test)")
    st.dataframe(preds, use_container_width=True)

    st.subheader("Métricas (comparativa)")
    st.dataframe(metrics, use_container_width=True)

# --- Optional: show generated PNGs filtered by country ---
if show_figures_folder:
    st.subheader(f"Figuras generadas — {geo} ({bloque_label})")
    if fig_subdir.exists():
        pngs = sorted(fig_subdir.glob(f"*_{geo_lower}_*.png"))
        if not pngs:
            st.info(f"No hay PNGs para {geo} en {fig_subdir.relative_to(REPO_ROOT)}")
        else:
            for p in pngs:
                st.write(p.name)
                st.image(str(p), use_container_width=True)
    else:
        st.info(f"No existe la carpeta {fig_subdir.relative_to(REPO_ROOT)}")

st.caption("TFG_INFO — Dashboard (Streamlit).")
