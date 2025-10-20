import sys
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

pib_path = ROOT / "data" / "processed" / "pib_trimestral.parquet"
df = pd.read_parquet(pib_path)

print("\n--- Información general ---")
print(df.info())
print("\n--- Primeras filas ---")
print(df.head())

plt.figure(figsize=(10, 5))
plt.plot(df.index.astype(str), df["pib_var"], marker="o")
plt.title("Variación trimestral del PIB español (%)")
plt.xlabel("Trimestre")
plt.ylabel("PIB var.")
plt.grid(True)

plt.gca().xaxis.set_major_locator(ticker.MultipleLocator(8))
plt.xticks(rotation=45, ha="right")

plt.tight_layout()

fig_dir = ROOT / "reports" / "figures"
fig_dir.mkdir(parents=True, exist_ok=True)
plt.savefig(fig_dir / "pib_series.png", dpi=150)
plt.show()

print("\n--- Estadísticos descriptivos")
print(df["pib_var"].describe())


def naive_forecast(series, h=1):
    return series.shift(1)


df["pib_pred_naive"] = naive_forecast(df["pib_var"])
df["error_abs"] = (df["pib_var"] - df["pib_pred_naive"]).abs()

mae = df["error_abs"].iloc[1:].mean()
print(f"\nMAE del modelo naive (h=1 trimestre): {mae:.3f}")

out_path = ROOT / "data" / "processed" / "pib_naive_eval.parquet"
df.to_parquet(out_path)
print(f"Resultados guardados en: {out_path}")
