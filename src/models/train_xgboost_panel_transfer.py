"""
Bloque 3 — Transferencia entre países.

Para cada una de las 3 combinaciones posibles, entrena un XGBoost ampliado
(ext) con los datos de dos países y evalúa sobre el tercero, que no ha
participado en el entrenamiento.

Combinaciones:
  ES + FR → evalúa en IT
  ES + IT → evalúa en FR
  FR + IT → evalúa en ES
"""
from pathlib import Path
import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt


DATA_PATH = Path("../../data/processed/dataset_panel_v1.parquet")
OUT_DIR   = Path("../../reports")
FIG_DIR   = OUT_DIR / "figures" / "bloque3"
OUT_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_START = "2009Q2"
TRAIN_END   = "2023Q3"
TEST_START  = "2023Q4"
TEST_END    = "2025Q3"

TARGET = "gdp_qoq_pct"

FEATURES_EXT = [
    "gdp_qoq_pct_l1", "gdp_qoq_pct_l2",
    "unemployment_rate_l1", "inflation_qoq_pct_l1",
    "ipi_qoq_pct_l1", "retail_qoq_pct_l1",
]

# Las 3 combinaciones: (países de entrenamiento, país de evaluación)
TRANSFER_CONFIGS = [
    (["ES", "FR"], "IT"),
    (["ES", "IT"], "FR"),
    (["FR", "IT"], "ES"),
]


def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))


def plot_real_vs_pred(periods, y_true, y_pred, title, out_path):
    plt.figure()
    plt.plot(periods, y_true, marker="o", label="Real")
    plt.plot(periods, y_pred, marker="o", label="Predicción")
    plt.title(title)
    plt.xlabel("Periodo (trimestre)")
    plt.ylabel("Crecimiento PIB QoQ (%)")
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def plot_errors(periods, errors, title, out_path):
    plt.figure()
    plt.plot(periods, errors, marker="o", label="Error (Real - Pred)")
    plt.axhline(0, linestyle="--")
    plt.title(title)
    plt.xlabel("Periodo (trimestre)")
    plt.ylabel("Error (p.p.)")
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def run_transfer(df, train_geos, test_geo):
    """
    Entrena con los países de train_geos y evalúa en test_geo.
    Devuelve un dict con las métricas del experimento.
    """
    label = "+".join(train_geos)
    model_name = f"panel_transfer_ext"

    use_cols = ["geo", "period", TARGET] + FEATURES_EXT
    d = df[use_cols].dropna().copy()

    # Entrenamiento: dos países, periodo de train
    train = d[
        d["geo"].isin(train_geos) &
        (d["period"] >= TRAIN_START) &
        (d["period"] <= TRAIN_END)
    ]

    # Evaluación: tercer país, solo periodo de test
    test = d[
        (d["geo"] == test_geo) &
        (d["period"] >= TEST_START) &
        (d["period"] <= TEST_END)
    ]

    print(f"\n=== Transferencia: entrena [{label}] → evalúa [{test_geo}] ===")
    print(f"  Train: {TRAIN_START}–{TRAIN_END}  (n={len(train)}, países: {sorted(train['geo'].unique())})")
    print(f"  Test : {TEST_START}–{TEST_END}   (n={len(test)},  país: {test_geo})")

    X_train = train[FEATURES_EXT]
    y_train = train[TARGET]

    model = XGBRegressor(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=3,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=42,
    )
    model.fit(X_train, y_train)

    X_test = test[FEATURES_EXT]
    y_test = test[TARGET]
    y_pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    r   = rmse(y_test, y_pred)
    print(f"  MAE={mae:.4f}  RMSE={r:.4f}")

    # CSV de predicciones — misma convención de nombres que bloques anteriores
    out = test[["period", TARGET]].copy()
    out["y_pred"] = y_pred
    out["error"]  = out[TARGET] - out["y_pred"]
    out = out.rename(columns={TARGET: "y_true"})
    out_csv = OUT_DIR / f"predictions_{model_name}_{test_geo.lower()}.csv"
    out.to_csv(out_csv, index=False)

    # Gráficas
    periods = out["period"].astype(str).tolist()
    plot_real_vs_pred(
        periods, out["y_true"].values, out["y_pred"].values,
        title=f"PIB QoQ (%) — Real vs Predicción (Transfer [{label}]→{test_geo}, test)",
        out_path=FIG_DIR / f"panel_transfer_ext_{test_geo.lower()}_real_vs_pred.png",
    )
    plot_errors(
        periods, out["error"].values,
        title=f"Errores de predicción — Transfer [{label}]→{test_geo} (test)",
        out_path=FIG_DIR / f"panel_transfer_ext_{test_geo.lower()}_errors.png",
    )

    return {
        "model": model_name,
        "train_geos": label,
        "geo": test_geo,
        "mae": mae,
        "rmse": r,
        "n_features": len(FEATURES_EXT),
    }


def main():
    df = pd.read_parquet(DATA_PATH).copy()

    if df["period"].dtype == object:
        df["period"] = pd.PeriodIndex(df["period"], freq="Q")

    df = df.sort_values(["geo", "period"]).reset_index(drop=True)

    all_metrics = []
    for train_geos, test_geo in TRANSFER_CONFIGS:
        result = run_transfer(df, train_geos, test_geo)
        all_metrics.append(result)

    metrics_df = pd.DataFrame(all_metrics)
    metrics_path = OUT_DIR / "metrics_panel_transfer_compare.csv"
    metrics_df.to_csv(metrics_path, index=False)
    print("\nMétricas guardadas en:", metrics_path)
    print(metrics_df.to_string(index=False))


if __name__ == "__main__":
    main()
