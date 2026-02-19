from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

DATASET_PATH = Path("../../data/processed/dataset_v1.parquet")
FIG_DIR = Path("../../reports/figures")
OUT_PRED_CSV = Path("../../reports/predictions_linear_multivar_es.csv")


def mae(y_true, y_pred) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.mean(np.abs(y_true - y_pred)))


def rmse(y_true, y_pred) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def save_plot_real_vs_pred(period_str, y_true, y_pred, out_path: Path):
    plt.figure(figsize=(9, 4))
    plt.plot(period_str, y_true, marker="o", label="Real")
    plt.plot(period_str, y_pred, marker="o", label="Predicción (Lineal multivar.)")
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
    plt.title("Errores de predicción — Lineal multivariante (test)")
    plt.xlabel("Periodo (trimestre)")
    plt.ylabel("Error (p.p.)")
    plt.xticks(rotation=45, ha="right")
    plt.grid(True, alpha=0.3)
    plt.legend()
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

    # Lags (t-1) para evitar info contemporánea
    df["unemp_l1"] = df["unemployment_rate"].shift(1)
    df["infl_l1"] = df["inflation_qoq_pct"].shift(1)

    df = df.dropna(subset=["gdp_qoq_pct", "unemp_l1", "infl_l1"]).reset_index(drop=True)

    if len(df) <= test_horizon + 20:
        raise ValueError(
            f"Pocas observaciones ({len(df)}) para test_horizon={test_horizon}."
        )

    train_df = df.iloc[:-test_horizon].copy()
    test_df = df.iloc[-test_horizon:].copy()

    X_train = train_df[["unemp_l1", "infl_l1"]].astype("float64")
    y_train = train_df["gdp_qoq_pct"].astype("float64")

    X_test = test_df[["unemp_l1", "infl_l1"]].astype("float64")
    y_test = test_df["gdp_qoq_pct"].astype("float64")

    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    m_mae = mae(y_test, y_pred)
    m_rmse = rmse(y_test, y_pred)

    print("\n=== Linear Regression Multivariante (Spain) ===")
    print(f"Dataset: {DATASET_PATH.as_posix()}")
    print("Features: unemp_l1, infl_l1")
    print(f"Train: {train_df['period'].iloc[0]} -> {train_df['period'].iloc[-1]}")
    print(f"Test : {test_df['period'].iloc[0]} -> {test_df['period'].iloc[-1]}  (h={test_horizon})")
    print(f"MAE={m_mae:.4f}  RMSE={m_rmse:.4f}")

    print("\nCoeficientes:")
    print(f"Intercept: {model.intercept_:.6f}")
    print(f"Beta(unemp_l1): {model.coef_[0]:.6f}")
    print(f"Beta(infl_l1):  {model.coef_[1]:.6f}")

    out = test_df[["period", "gdp_qoq_pct", "unemp_l1", "infl_l1"]].copy()
    out = out.rename(columns={"gdp_qoq_pct": "y_true"})
    out["y_pred"] = y_pred
    out["error"] = out["y_true"] - out["y_pred"]

    print("\n--- Predicciones (test) ---")
    print(out.to_string(index=False))

    out.to_csv(OUT_PRED_CSV, index=False)
    print(f"\nPredicciones guardadas en: {OUT_PRED_CSV.resolve()}")

    period_str = out["period"].astype(str).tolist()
    fig1 = FIG_DIR / "linear_multivar_es_real_vs_pred.png"
    fig2 = FIG_DIR / "linear_multivar_es_errors.png"

    save_plot_real_vs_pred(period_str, out["y_true"].values, out["y_pred"].values, fig1)
    save_plot_errors(period_str, out["error"].values, fig2)

    print(f"Figura guardada: {fig1.resolve()}")
    print(f"Figura guardada: {fig2.resolve()}")


if __name__ == "__main__":
    main(test_horizon=8)
