import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# statsmodels (ARIMA)
from statsmodels.tsa.arima.model import ARIMA

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

pib_path = ROOT / "data" / "processed" / "pib_trimestral.parquet"
df  = pd.read_parquet(pib_path)

y = df["pib_var"].dropna().astype(float)

print(f"Observaciones: {len(y)} | Desde {y.index.min()} hasta {y.index.max()}")
print("Primeros valores:")
print(y.head)

TEST_SIZE = 12
y_train = y.iloc[:-TEST_SIZE]
y_test = y.iloc[-TEST_SIZE:]

print(f"\nTrain: {y_train.index.min()} -> {y_train.index.max()} ({len(y_train)})")
print(f"Test: {y_test.index.min()} -> {y_test.index.max()} ({len(y_test)})")

naive_pred = y.shift(1).loc[y_test.index]
naive_mae = np.mean(np.abs(y_test - naive_pred))
print(f"\nMAE Naive (test, h=1): {naive_mae:.3f}")

order = (1, 0, 1)

history = list(y_train.values)
arima_preds = []

for t in range(len(y_test)):
    model = ARIMA(history, order=order)
    model_fit = model.fit()

    yhat = model_fit.forecast(steps=1)[0]
    arima_preds.append(yhat)

    history.append(y_test.values[t])

arima_pred = pd.Series(arima_preds, index=y_test.index)
arima_mae =  np.mean(np.abs(y_test - arima_pred))
print(f"MAE ARIMA{order} (test, walk-forward h=1): {arima_mae:.3f}")

out_df = pd.DataFrame({
    "y_real": y_test,
    "y_naive": naive_pred,
    "y_arima": arima_pred,
})

out_path = ROOT / "data" / "processed" / "pib_arima_eval.parquet"
out_df.to_parquet(out_path)
print(f"\nResultados guardados en {out_path}")

plt.figure(figsize=(10, 5))
plt.plot(out_df.index.astype(str), out_df["y_real"], marker="o", label="Real")
plt.plot(out_df.index.astype(str), out_df["y_naive"], marker="o", label="Naive")
plt.plot(out_df.index.astype(str), out_df["y_arima"], marker="o", label="ARIMA{order}")
plt.title("PIB: Real vs Predicción (últimos 12 trimestres)")
plt.xlabel("Trimestre")
plt.ylabel("Variación del PIB (%)")
plt.grid(True, alpha=0.3)
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.legend()

fig_dir = ROOT / "reports" / "figures"
fig_dir.mkdir(parents=True, exist_ok=True)
fig_path = fig_dir / "pib_arima_vs_naive.png"
plt.savefig(fig_path, dpi=150)
plt.show()

print(f"Gráfico guardado en: {fig_path}")