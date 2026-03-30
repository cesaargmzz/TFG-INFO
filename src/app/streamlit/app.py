import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from pathlib import Path

# --- Paths ---
REPO_ROOT   = Path(__file__).resolve().parents[3]
REPORTS_DIR = REPO_ROOT / "reports"

GEO_LABELS = {"ES": "España", "FR": "Francia", "IT": "Italia"}

BLOQUE_OPTIONS = {
    "Bloque 0 — Por país":                  0,
    "Bloque 1 — Panel combinado":            1,
    "Bloque 2 — Panel con geo como feature": 2,
    "Bloque 3 — Transferencia entre países": 3,
}

BLOQUE_DESC = {
    0: ("**Bloque 0 — Modelos por país**\n\n"
        "Cada modelo se entrena y evalúa de forma independiente usando únicamente los datos "
        "del país seleccionado (~58 trimestres). Es la línea de referencia individual: "
        "muestra qué rendimiento es posible cuando el modelo no comparte información entre economías."),
    1: ("**Bloque 1 — Panel combinado**\n\n"
        "Un único modelo XGBoost se entrena con los datos de España, Francia e Italia "
        "combinados (~174 trimestres) y se evalúa por separado en cada país. "
        "El mayor volumen de datos reduce el sobreajuste observado en el Bloque 0, "
        "especialmente en el modelo ampliado."),
    2: ("**Bloque 2 — Panel con variable geo como feature**\n\n"
        "Igual que el Bloque 1, pero añadiendo variables indicadoras de país "
        "(geo_ES, geo_FR, geo_IT) al vector de entrada del modelo. "
        "El modelo puede así aprender umbrales distintos por economía. "
        "Sin lags del PIB, estas dummies se sobreexplotan y degradan las predicciones; "
        "con lags, el modelo prácticamente las ignora."),
    3: ("**Bloque 3 — Transferencia entre países**\n\n"
        "El modelo se entrena con dos países y se evalúa en el tercero, "
        "que no ha participado en el entrenamiento. "
        "Permite explorar si los patrones macroeconómicos son transferibles entre economías. "
        "Solo se usa la configuración ampliada (ext)."),
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

def load_all_metrics():
    """Carga y combina métricas de todos los bloques para la tabla resumen."""
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


# --- Sidebar ---
st.sidebar.header("Controles")

bloque_label = st.sidebar.selectbox("Bloque experimental", list(BLOQUE_OPTIONS.keys()), index=0)
bloque = BLOQUE_OPTIONS[bloque_label]

geo = st.sidebar.selectbox(
    "País", list(GEO_LABELS.keys()),
    format_func=lambda g: f"{g} — {GEO_LABELS[g]}", index=0,
)

if bloque == 0:
    model_options = {"XGBoost (base)": "base", "XGBoost (ampliado)": "ext"}
elif bloque == 1:
    model_options = {"Panel combinado (base)": "base", "Panel combinado (ampliado)": "ext"}
elif bloque == 2:
    model_options = {"Panel geo (base)": "base", "Panel geo (ampliado)": "ext"}
else:
    model_options = {"Transfer (ampliado)": "ext"}

model_label = st.sidebar.selectbox("Modelo", list(model_options.keys()), index=0)
model_key   = model_options[model_label]

st.sidebar.divider()
show_tables  = st.sidebar.checkbox("Mostrar tablas de datos", value=False)
show_summary = st.sidebar.checkbox("Mostrar tabla resumen global", value=False)


# --- Dynamic paths ---
geo_lower = geo.lower()

if bloque == 0:
    metrics_path   = REPORTS_DIR / f"metrics_xgboost_{geo_lower}_compare.csv"
    pred_base_path = REPORTS_DIR / f"predictions_xgboost_base_{geo_lower}.csv"
    pred_ext_path  = REPORTS_DIR / f"predictions_xgboost_ext_{geo_lower}.csv"
elif bloque == 1:
    metrics_path   = REPORTS_DIR / "metrics_panel_combined_compare.csv"
    pred_base_path = REPORTS_DIR / f"predictions_panel_combined_base_{geo_lower}.csv"
    pred_ext_path  = REPORTS_DIR / f"predictions_panel_combined_ext_{geo_lower}.csv"
elif bloque == 2:
    metrics_path   = REPORTS_DIR / "metrics_panel_geo_compare.csv"
    pred_base_path = REPORTS_DIR / f"predictions_panel_geo_base_{geo_lower}.csv"
    pred_ext_path  = REPORTS_DIR / f"predictions_panel_geo_ext_{geo_lower}.csv"
else:
    metrics_path   = REPORTS_DIR / "metrics_panel_transfer_compare.csv"
    pred_base_path = None
    pred_ext_path  = REPORTS_DIR / f"predictions_panel_transfer_ext_{geo_lower}.csv"


# --- Load data ---
metrics_raw = load_csv(metrics_path)
if bloque in (1, 2, 3) and "geo" in metrics_raw.columns:
    metrics = metrics_raw[metrics_raw["geo"] == geo].reset_index(drop=True)
else:
    metrics = metrics_raw

pred_base = safe_period_str(load_csv(pred_base_path)) if pred_base_path else None
pred_ext  = safe_period_str(load_csv(pred_ext_path))

if model_key == "base" and pred_base is not None:
    preds       = pred_base.copy()
    metrics_row = metrics[metrics["model"].str.contains("base", case=False, na=False)]
else:
    preds       = pred_ext.copy()
    metrics_row = metrics[metrics["model"].str.contains("ext", case=False, na=False)]

if metrics_row.empty:
    metrics_row = metrics.head(1)

mae_val    = float(metrics_row["mae"].iloc[0])      if "mae"        in metrics_row.columns else None
rmse_val   = float(metrics_row["rmse"].iloc[0])     if "rmse"       in metrics_row.columns else None
n_features = int(metrics_row["n_features"].iloc[0]) if "n_features" in metrics_row.columns else None


# ══════════════════════════════════════════════════════
# LAYOUT PRINCIPAL
# ══════════════════════════════════════════════════════

st.title(f"Dashboard — PIB QoQ · {geo} ({GEO_LABELS[geo]}) · {bloque_label}")

# --- Descripción del bloque ---
with st.expander("¿Qué es este bloque?", expanded=True):
    st.markdown(BLOQUE_DESC[bloque])

st.divider()

# --- KPIs ---
kpi1, kpi2, kpi3, kpi4 = st.columns(4)
kpi1.metric("Modelo activo", model_label)
kpi2.metric("MAE (p.p.)",    f"{mae_val:.3f}"  if mae_val  is not None else "—")
kpi3.metric("RMSE (p.p.)",   f"{rmse_val:.3f}" if rmse_val is not None else "—")
kpi4.metric("Nº variables",  str(n_features)   if n_features is not None else "—")

if bloque == 3 and "train_geos" in metrics_row.columns and not metrics_row.empty:
    train_geos_val = metrics_row["train_geos"].iloc[0]
    st.info(f"Entrenado con **{train_geos_val}** — evaluado en **{geo}** (país no visto en entrenamiento)")

st.divider()

# --- Gráfica principal: base vs ampliado superpuestos (cuando hay ambos) ---
if pred_base is not None and {"period", "y_true", "y_pred"}.issubset(pred_base.columns):
    st.subheader("Real vs Predicción — comparación base y ampliado")
    fig_cmp = go.Figure()
    fig_cmp.add_trace(go.Scatter(
        x=pred_ext["period"], y=pred_ext["y_true"],
        mode="lines+markers", name="Real",
        line=dict(color="black", width=2),
    ))
    fig_cmp.add_trace(go.Scatter(
        x=pred_base["period"], y=pred_base["y_pred"],
        mode="lines+markers", name=f"Base ({list(model_options.keys())[0]})",
        line=dict(dash="dot"),
    ))
    fig_cmp.add_trace(go.Scatter(
        x=pred_ext["period"], y=pred_ext["y_pred"],
        mode="lines+markers", name=f"Ampliado ({list(model_options.keys())[1]})",
    ))
    fig_cmp.update_layout(
        xaxis_title="Trimestre", yaxis_title="PIB QoQ (%)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        hovermode="x unified",
    )
    st.plotly_chart(fig_cmp, use_container_width=True)
    st.divider()

# --- Gráficas individuales del modelo seleccionado ---
col1, col2 = st.columns(2)

with col1:
    st.subheader("Real vs Predicción (modelo seleccionado)")
    if {"period", "y_true", "y_pred"}.issubset(preds.columns):
        fig = px.line(
            preds, x="period", y=["y_true", "y_pred"],
            markers=True,
            labels={"value": "PIB QoQ (%)", "period": "Trimestre", "variable": ""},
            color_discrete_map={"y_true": "black", "y_pred": "#1f77b4"},
        )
        fig.update_layout(hovermode="x unified",
                          legend=dict(orientation="h", yanchor="bottom", y=1.02))
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("CSV sin columnas esperadas: period, y_true, y_pred")

with col2:
    st.subheader("Error de predicción trimestre a trimestre")
    if {"period", "error"}.issubset(preds.columns):
        fig2 = px.bar(
            preds, x="period", y="error",
            labels={"error": "Error (p.p.)", "period": "Trimestre"},
            color="error",
            color_continuous_scale=["#d62728", "#ffffff", "#1f77b4"],
            color_continuous_midpoint=0,
        )
        fig2.add_hline(y=0, line_dash="dash", line_color="black")
        fig2.update_layout(coloraxis_showscale=False, hovermode="x unified")
        st.plotly_chart(fig2, use_container_width=True)
    else:
        st.warning("CSV sin columnas esperadas: period, error")

# --- Tablas opcionales ---
if show_tables:
    st.divider()
    st.subheader("Predicciones — conjunto de test")
    st.dataframe(preds, use_container_width=True)
    st.subheader("Métricas — comparativa del bloque")
    st.dataframe(metrics, use_container_width=True)

# --- Tabla resumen global ---
if show_summary:
    st.divider()
    st.subheader("Resumen global — MAE por bloque y país (modelo ampliado)")
    all_metrics = load_all_metrics()
    if not all_metrics.empty:
        ext_only = all_metrics[all_metrics["model"].str.contains("ext|transfer", case=False, na=False)]
        pivot = ext_only.pivot_table(
            index="bloque", columns="geo", values="mae", aggfunc="mean"
        ).round(3)
        pivot.index = [
            "B0 — Por país", "B1 — Panel combinado",
            "B2 — Panel geo", "B3 — Transferencia",
        ][:len(pivot)]
        st.dataframe(pivot.style.highlight_min(axis=0, color="#c6efce")
                                .highlight_max(axis=0, color="#ffc7ce"),
                     use_container_width=True)
        st.caption("Verde = mejor MAE por país. Rojo = peor MAE por país.")
    else:
        st.info("No se pudieron cargar las métricas de todos los bloques.")

st.divider()
st.caption("TFG — Análisis comparativo de modelos de ML para la estimación del PIB en la zona euro.")
