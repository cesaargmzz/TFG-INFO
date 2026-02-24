import pandas as pd
import plotly.express as px
import streamlit as st
from pathlib import Path

# --- Paths (repo root assumed) ---
REPO_ROOT = Path(__file__).resolve().parents[3]  # src/app/streamlit/app.py -> repo root
REPORTS_DIR = REPO_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

METRICS_PATH = REPORTS_DIR / "metrics_xgboost_es_compare.csv"
PRED_BASE_PATH = REPORTS_DIR / "predictions_xgboost_base.csv"
PRED_EXT_PATH = REPORTS_DIR / "predictions_xgboost_ext.csv"

st.set_page_config(page_title="TFG Dashboard — PIB QoQ (España)", layout="wide")

st.title("Dashboard — Predicción del PIB QoQ (España)")
st.caption("MVP: visualización interactiva de resultados de modelos (CSV generados por el pipeline).")

# --- Helpers ---
def load_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        st.error(f"No se encontró el archivo: {path}")
        st.stop()
    return pd.read_csv(path)

def safe_period_str(df: pd.DataFrame) -> pd.DataFrame:
    # Asegura que period sea string para plotly
    if "period" in df.columns:
        df["period"] = df["period"].astype(str)
    return df

# --- Load data ---
metrics = load_csv(METRICS_PATH)
pred_base = safe_period_str(load_csv(PRED_BASE_PATH))
pred_ext = safe_period_str(load_csv(PRED_EXT_PATH))

# --- Sidebar controls ---
st.sidebar.header("Controles")

model_options = {
    "XGBoost (base)": "base",
    "XGBoost (ampliado)": "ext",
}

model_label = st.sidebar.selectbox("Modelo", list(model_options.keys()), index=0)
model_key = model_options[model_label]

show_tables = st.sidebar.checkbox("Mostrar tablas", value=True)
show_figures_folder = st.sidebar.checkbox("Mostrar carpeta de figuras (PNG)", value=False)

# --- Select predictions ---
if model_key == "base":
    preds = pred_base.copy()
    metrics_row = metrics[metrics["model"].str.contains("base", case=False, na=False)]
else:
    preds = pred_ext.copy()
    metrics_row = metrics[metrics["model"].str.contains("ext", case=False, na=False)]

# Fallback si el nombre exacto no coincide
if metrics_row.empty:
    metrics_row = metrics.head(1)

mae = float(metrics_row["mae"].iloc[0]) if "mae" in metrics_row.columns else None
rmse = float(metrics_row["rmse"].iloc[0]) if "rmse" in metrics_row.columns else None
n_features = int(metrics_row["n_features"].iloc[0]) if "n_features" in metrics_row.columns else None

# --- KPIs ---
kpi1, kpi2, kpi3 = st.columns(3)
kpi1.metric("Modelo", model_label)
kpi2.metric("MAE (p.p.)", f"{mae:.3f}" if mae is not None else "—")
kpi3.metric("RMSE (p.p.)", f"{rmse:.3f}" if rmse is not None else "—")

if n_features is not None:
    st.caption(f"Nº de variables: {n_features}")

st.divider()

# --- Plots ---
# Real vs Pred
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

# --- Optional: show generated PNGs ---
if show_figures_folder:
    st.subheader("Figuras generadas (reports/figures)")
    if FIGURES_DIR.exists():
        pngs = sorted(FIGURES_DIR.glob("*.png"))
        if not pngs:
            st.info("No hay PNGs en reports/figures")
        else:
            for p in pngs:
                st.write(p.name)
                st.image(str(p), use_container_width=True)
    else:
        st.info("No existe la carpeta reports/figures")

st.caption("TFG_INFO — Dashboard MVP (Streamlit).")