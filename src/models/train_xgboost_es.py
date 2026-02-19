from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import mean_absolute_error, mean_squared_error
from xgboost import XGBRegressor

DATASET_PATH = Path("../../data/processed/dataset_v1.parquet")
FIG_DIR = Path("../../reports/figures")
OUT_PRED_CSV = Path("../../reports/predictions_xgboost_es.csv")


def rmse(y_true, y_pred) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def save_plot_real_vs_pred(period_str, y_true, y_pred, out_path: Path):
    plt.figure(figsize=(9, 4))
    plt.plot(period_str, y_true, marker="o", label="Real")
    plt.plot(period_str, y_pred, marker="o", label="Predicción (XGBoost)")
    plt.title("PIB QoQ (%) — Real vs Predicción (test)")
    plt.xlabel("Periodo (trimestre)")
    plt.ylabel("Crecimiento PIB QoQ (%)")
    plt.xticks(rotation=45, ha="right")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def save_plot_errors(period_str, errors, out_path: Path):
    plt.figure(figsize=(9, 4))
    plt.plot(period_str, errors, marker="o", label="Error (Real - Pred)")
    plt.axhline(0, linestyle="--", linewidth=1)
    plt.title("Errores de predicción — XGBoost (test)")
    plt.xlabel("Periodo (trimestre)")
    plt.ylabel("Error (p.p.)")
    plt.xticks(rotation=45, ha="right")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def save_plot_feature_importance(feature_names, importances, out_path: Path):
    order = np.argsort(importances)[::-1]
    names_sorted = [feature_names[i] for i in order]
    imps_sorted = importances[order]

    plt.figure(figsize=(7, 4))
    plt.bar(names_sorted, imps_sorted)
    plt.title("Importancia de variables (XGBoost)")
    plt.xlabel("Variable")
    plt.ylabel("Importancia (gain)")
    plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def main(test_horizon: int = 8):
    if not DATASET_PATH.exists():
        raise FileNotFoundError(f"No encuentro el dataset: {DATASET_PATH.resolve()}")

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    OUT_PRED_CSV.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(DATASET_PATH)

    # Solo España
    df = df[df["geo"] == "ES"].copy()
    df = df.sort_values("period").reset_index(drop=True)

    # Lags
    df["unemp_l1"] = df["unemployment_rate"].shift(1)
    df["infl_l1"] = df["inflation_qoq_pct"].shift(1)

    feature_cols = ["unemp_l1", "infl_l1"]
    target_col = "gdp_qoq_pct"

    df = df.dropna(subset=[target_col] + feature_cols).reset_index(drop=True)

    # Sanity checks (MUY IMPORTANTES)
    for col in [target_col] + feature_cols:
        if not np.isfinite(df[col].to_numpy()).all():
            bad = df[~np.isfinite(df[col])][["period", col]].head(10)
            raise ValueError(f"Valores no finitos detectados en {col}:\n{bad}")

    print("\n[Sanity] Target stats (España):")
    print(df[target_col].describe())

    # Split temporal correcto (últimos h trimestres = test)
    train_df = df.iloc[:-test_horizon].copy()
    test_df = df.iloc[-test_horizon:].copy()

    print("\n[Split]")
    print(f"Train: {train_df['period'].iloc[0]} -> {train_df['period'].iloc[-1]} (n={len(train_df)})")
    print(f"Test : {test_df['period'].iloc[0]} -> {test_df['period'].iloc[-1]} (n={len(test_df)})")

    X_train = train_df[feature_cols].astype("float64")
    y_train = train_df[target_col].astype("float64")

    X_test = test_df[feature_cols].astype("float64")
    y_test = test_df[target_col].astype("float64")

    model = XGBRegressor(
        n_estimators=400,
        learning_rate=0.05,
        max_depth=3,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_alpha=0.0,
        reg_lambda=1.0,
        objective="reg:squarederror",
        random_state=42,
    )

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    m_mae = float(mean_absolute_error(y_test, y_pred))
    m_rmse = rmse(y_test, y_pred)

    print("\n=== XGBoost (Spain) ===")
    print(f"Dataset: {DATASET_PATH.as_posix()}")
    print(f"Features: {', '.join(feature_cols)}")
    print(f"MAE={m_mae:.4f}  RMSE={m_rmse:.4f}")

    out = test_df[["period", target_col] + feature_cols].copy()
    out = out.rename(columns={target_col: "y_true"})
    out["y_pred"] = y_pred
    out["error"] = out["y_true"] - out["y_pred"]

    print("\n--- Predicciones (test) ---")
    print(out.to_string(index=False))

    out.to_csv(OUT_PRED_CSV, index=False)
    print(f"\nPredicciones guardadas en: {OUT_PRED_CSV.resolve()}")

    # Figuras
    period_str = out["period"].astype(str).tolist()

    fig1 = FIG_DIR / "xgboost_es_real_vs_pred.png"
    fig2 = FIG_DIR / "xgboost_es_errors.png"
    fig3 = FIG_DIR / "xgboost_es_feature_importance.png"

    save_plot_real_vs_pred(period_str, out["y_true"].values, out["y_pred"].values, fig1)
    save_plot_errors(period_str, out["error"].values, fig2)

    # Importancia
    booster = model.get_booster()
    score = booster.get_score(importance_type="gain")
    importances = np.array([score.get(f"f{i}", 0.0) for i in range(len(feature_cols))], dtype=float)

    save_plot_feature_importance(feature_cols, importances, fig3)

    print(f"Figura guardada: {fig1.resolve()}")
    print(f"Figura guardada: {fig2.resolve()}")
    print(f"Figura guardada: {fig3.resolve()}")


if __name__ == "__main__":
    main(test_horizon=8)
