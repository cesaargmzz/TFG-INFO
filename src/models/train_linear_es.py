from pathlib import Path
import pandas as pd
import numpy as np

from sklearn.linear_model import LinearRegression


DATASET_PATH = Path("../../data/processed/dataset_es_v1.parquet")


def mae(y_true, y_pred) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.mean(np.abs(y_true - y_pred)))


def rmse(y_true, y_pred) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def main(test_horizon: int = 8, use_lagged_feature: bool = True):
    if not DATASET_PATH.exists():
        raise FileNotFoundError(f"No encuentro el dataset: {DATASET_PATH.resolve()}")

    df = pd.read_parquet(DATASET_PATH).sort_values("period").reset_index(drop=True)

    # Feature engineering mínimo: usar desempleo rezagado para evitar "mirar el mismo trimestre"
    if use_lagged_feature:
        df["unemployment_rate_l1"] = df["unemployment_rate"].shift(1)
        feature_col = "unemployment_rate_l1"
    else:
        feature_col = "unemployment_rate"

    # Quitar NaNs (por el lag)
    df = df.dropna(subset=["gdp_qoq_pct", feature_col]).reset_index(drop=True)

    if len(df) <= test_horizon + 20:
        raise ValueError("Muy pocas observaciones para un train/test fiable con ese horizonte.")

    train_df = df.iloc[:-test_horizon].copy()
    test_df = df.iloc[-test_horizon:].copy()

    X_train = train_df[[feature_col]].astype("float64")
    y_train = train_df["gdp_qoq_pct"].astype("float64")

    X_test = test_df[[feature_col]].astype("float64")
    y_test = test_df["gdp_qoq_pct"].astype("float64")

    # Modelo
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Métricas
    m_mae = mae(y_test, y_pred)
    m_rmse = rmse(y_test, y_pred)

    # Report
    print("\n=== Linear Regression (Spain) ===")
    print(f"Dataset: {DATASET_PATH.as_posix()}")
    print(f"Feature: {feature_col}  (lagged={use_lagged_feature})")
    print(f"Train: {train_df['period'].iloc[0]} -> {train_df['period'].iloc[-1]}")
    print(f"Test : {test_df['period'].iloc[0]} -> {test_df['period'].iloc[-1]}  (h={test_horizon})")
    print(f"MAE={m_mae:.4f}  RMSE={m_rmse:.4f}")

    print("\nCoeficientes:")
    print(f"Intercept: {model.intercept_:.6f}")
    print(f"Beta({feature_col}): {model.coef_[0]:.6f}")

    # Predicciones
    out = test_df[["period", "gdp_qoq_pct", feature_col]].copy()
    out = out.rename(columns={"gdp_qoq_pct": "y_true", feature_col: "x_feature"})
    out["y_pred"] = y_pred
    out["error"] = out["y_true"] - out["y_pred"]

    print("\n--- Predicciones (test) ---")
    print(out.to_string(index=False))


if __name__ == "__main__":
    # test_horizon=8 => últimos 2 años (8 trimestres)
    main(test_horizon=8, use_lagged_feature=True)
