"""
Bloque 1 — Modelo panel combinado (ES + FR + IT).

Entrena un único XGBoost (base y ext) con los datos de los 3 países
combinados y evalúa por separado en cada uno.
"""
from pathlib import Path
import argparse
import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt


DATA_PATH = Path("../../data/processed/dataset_panel_v1.parquet")
OUT_DIR   = Path("../../reports")
FIG_DIR   = OUT_DIR / "figures" / "bloque1"
OUT_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_START = "2009Q2"
TRAIN_END   = "2023Q3"
TEST_START  = "2023Q4"
TEST_END    = "2025Q3"

TARGET = "gdp_qoq_pct"

FEATURES_BASE = ["unemployment_rate_l1", "inflation_qoq_pct_l1"]

FEATURES_EXT = [
    "gdp_qoq_pct_l1", "gdp_qoq_pct_l2",
    "unemployment_rate_l1", "inflation_qoq_pct_l1",
    "ipi_qoq_pct_l1", "retail_qoq_pct_l1",
]

GEOS = ["ES", "FR", "IT"]


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


def train_combined(df, features, model_name):
    """Entrena un único modelo con los 3 países combinados."""
    use_cols = ["geo", "period", TARGET] + features
    d = df[use_cols].dropna().copy()

    train = d[(d["period"] >= TRAIN_START) & (d["period"] <= TRAIN_END)]
    test  = d[(d["period"] >= TEST_START)  & (d["period"] <= TEST_END)]

    X_train = train[features]
    y_train = train[TARGET]

    print(f"\n=== {model_name} — entrenamiento combinado ===")
    print("Features:", ", ".join(features))
    print(f"Train: {TRAIN_START} -> {TRAIN_END}  (n={len(train)}, países: {sorted(train['geo'].unique())})")
    print(f"Test : {TEST_START} -> {TEST_END}   (n={len(test)},  países: {sorted(test['geo'].unique())})")

    model = XGBRegressor(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=3,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=42,
    )
    model.fit(X_train, y_train)

    # Feature importance (gain) del modelo global
    slug = model_name.lower().replace(" ", "_")
    try:
        booster = model.get_booster()
        score = booster.get_score(importance_type="gain")
        imp = pd.DataFrame(
            {"feature": list(score.keys()), "gain": list(score.values())}
        ).sort_values("gain", ascending=False)

        plt.figure()
        plt.bar(imp["feature"], imp["gain"])
        plt.title(f"Importancia de variables (gain) — {model_name} (panel combinado)")
        plt.xlabel("Variable")
        plt.ylabel("Importancia (gain)")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(FIG_DIR / f"{slug}_combined_feature_importance.png", dpi=160)
        plt.close()
    except Exception as e:
        print("No se pudo guardar feature importance:", e)

    return model, test


def eval_by_country(model, test_all, features, model_name):
    """Evalúa el modelo por país y devuelve lista de métricas."""
    slug = model_name.lower().replace(" ", "_")
    results = []

    for geo in GEOS:
        test = test_all[test_all["geo"] == geo].copy()
        if test.empty:
            print(f"  Advertencia: no hay datos de test para {geo}")
            continue

        X_test = test[features]
        y_test = test[TARGET]
        y_pred = model.predict(X_test)

        mae = mean_absolute_error(y_test, y_pred)
        r   = rmse(y_test, y_pred)
        print(f"  {geo}:  MAE={mae:.4f}  RMSE={r:.4f}  (n={len(test)})")

        # CSV de predicciones
        out = test[["period", TARGET]].copy()
        out["y_pred"] = y_pred
        out["error"]  = out[TARGET] - out["y_pred"]
        out = out.rename(columns={TARGET: "y_true"})
        out_csv = OUT_DIR / f"predictions_{slug}_{geo.lower()}.csv"
        out.to_csv(out_csv, index=False)

        # Plots
        periods = out["period"].astype(str).tolist()
        plot_real_vs_pred(
            periods, out["y_true"].values, out["y_pred"].values,
            title=f"PIB QoQ (%) — Real vs Predicción ({model_name}, {geo}, test)",
            out_path=FIG_DIR / f"{slug}_{geo.lower()}_real_vs_pred.png",
        )
        plot_errors(
            periods, out["error"].values,
            title=f"Errores de predicción — {model_name} ({geo}, test)",
            out_path=FIG_DIR / f"{slug}_{geo.lower()}_errors.png",
        )

        results.append({"model": model_name, "geo": geo, "mae": mae, "rmse": r,
                        "n_features": len(features)})

    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(DATA_PATH), help="Ruta al dataset panel parquet")
    args = ap.parse_args()

    df = pd.read_parquet(Path(args.data)).copy()

    if df["period"].dtype == object:
        df["period"] = pd.PeriodIndex(df["period"], freq="Q")

    df = df.sort_values(["geo", "period"]).reset_index(drop=True)

    all_metrics = []

    for features, model_name in [
        (FEATURES_BASE, "panel_combined_base"),
        (FEATURES_EXT,  "panel_combined_ext"),
    ]:
        model, test_all = train_combined(df, features, model_name)
        metrics = eval_by_country(model, test_all, features, model_name)
        all_metrics.extend(metrics)

    metrics_df = pd.DataFrame(all_metrics)
    metrics_path = OUT_DIR / "metrics_panel_combined_compare.csv"
    metrics_df.to_csv(metrics_path, index=False)
    print("\nMétricas guardadas en:", metrics_path)
    print(metrics_df.to_string(index=False))


if __name__ == "__main__":
    main()
