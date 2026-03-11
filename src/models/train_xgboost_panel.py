import argparse
from pathlib import Path
import pandas as pd
import numpy as np

from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt


DATA_PATH = Path("../../data/processed/dataset_panel_v1.parquet")

OUT_DIR = Path("../../reports")
FIG_DIR = OUT_DIR / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

# Split temporal
TRAIN_START = "2009Q2"
TRAIN_END   = "2023Q3"
TEST_START  = "2023Q4"
TEST_END    = "2025Q3"

TARGET = "gdp_qoq_pct"

FEATURES_BASE = ["unemployment_rate_l1", "inflation_qoq_pct_l1"]

FEATURES_EXT = [
    "gdp_qoq_pct_l1", "gdp_qoq_pct_l2",
    "unemployment_rate_l1", "inflation_qoq_pct_l1",
    "ipi_qoq_pct_l1", "retail_qoq_pct_l1"
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


def train_and_eval(df, features, model_name, geo):
    # Drop NA solo en lo necesario
    use_cols = ["geo", "period", TARGET] + features
    d = df[use_cols].dropna().copy()

    # Split
    train = d[(d["period"] >= TRAIN_START) & (d["period"] <= TRAIN_END)]
    test  = d[(d["period"] >= TEST_START) & (d["period"] <= TEST_END)]

    X_train = train[features]
    y_train = train[TARGET]
    X_test = test[features]
    y_test = test[TARGET]

    print(f"\n=== {model_name} ({geo}) ===")
    print("Features:", ", ".join(features))
    print(f"Train: {TRAIN_START} -> {TRAIN_END} (n={len(train)})")
    print(f"Test : {TEST_START} -> {TEST_END} (n={len(test)})")

    model = XGBRegressor(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=3,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=42
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    r = rmse(y_test, y_pred)

    print(f"MAE={mae:.4f}  RMSE={r:.4f}")

    out = test[["period", TARGET]].copy()
    out["y_pred"] = y_pred
    out["error"] = out[TARGET] - out["y_pred"]
    out = out.rename(columns={TARGET: "y_true"})

    # Save preds — nombre incluye geo en minúsculas
    geo_lower = geo.lower()
    slug = model_name.lower().replace(" ", "_")
    out_csv = OUT_DIR / f"predictions_{slug}_{geo_lower}.csv"
    out.to_csv(out_csv, index=False)
    print("Predicciones guardadas en:", out_csv)

    # Plots
    periods = out["period"].astype(str).tolist()
    plot_real_vs_pred(
        periods, out["y_true"].values, out["y_pred"].values,
        title=f"PIB QoQ (%) — Real vs Predicción ({model_name}, {geo}, test)",
        out_path=FIG_DIR / f"{slug}_{geo_lower}_real_vs_pred.png"
    )
    plot_errors(
        periods, out["error"].values,
        title=f"Errores de predicción — {model_name} ({geo}, test)",
        out_path=FIG_DIR / f"{slug}_{geo_lower}_errors.png"
    )

    # Feature importance (gain)
    try:
        booster = model.get_booster()
        score = booster.get_score(importance_type="gain")
        imp = pd.DataFrame(
            {"feature": list(score.keys()), "gain": list(score.values())}
        ).sort_values("gain", ascending=False)

        plt.figure()
        plt.bar(imp["feature"], imp["gain"])
        plt.title(f"Importancia de variables (gain) — {model_name} ({geo})")
        plt.xlabel("Variable")
        plt.ylabel("Importancia (gain)")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(FIG_DIR / f"{slug}_{geo_lower}_feature_importance.png", dpi=160)
        plt.close()
    except Exception as e:
        print("No se pudo guardar feature importance:", e)

    return mae, r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--geo", default="ES", help="Código del país (ej: ES, FR, IT)")
    ap.add_argument("--data", default=str(DATA_PATH), help="Ruta al dataset panel parquet")
    args = ap.parse_args()

    geo = args.geo.upper()

    df = pd.read_parquet(Path(args.data)).copy()

    if df["period"].dtype == object:
        df["period"] = pd.PeriodIndex(df["period"], freq="Q")

    df = df.sort_values(["geo", "period"]).reset_index(drop=True)

    available = sorted(df["geo"].unique())
    if geo not in available:
        raise ValueError(f"geo='{geo}' no encontrado en el dataset. Disponibles: {available}")

    df = df[df["geo"] == geo].copy()

    mae_b, rmse_b = train_and_eval(df, FEATURES_BASE, "XGBoost_base", geo)
    mae_e, rmse_e = train_and_eval(df, FEATURES_EXT, "XGBoost_ext", geo)

    metrics = pd.DataFrame([
        {"model": "XGBoost_base", "geo": geo, "mae": mae_b, "rmse": rmse_b, "n_features": len(FEATURES_BASE)},
        {"model": "XGBoost_ext",  "geo": geo, "mae": mae_e, "rmse": rmse_e, "n_features": len(FEATURES_EXT)},
    ])

    metrics_path = OUT_DIR / f"metrics_xgboost_{geo.lower()}_compare.csv"
    metrics.to_csv(metrics_path, index=False)
    print("\nMétricas guardadas en:", metrics_path)
    print(metrics.to_string(index=False))


if __name__ == "__main__":
    main()
