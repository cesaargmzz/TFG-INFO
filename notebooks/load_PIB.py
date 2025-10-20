import pandas as pd

file = "../data/PIB.csv"

df = pd.read_csv(file, sep=";", encoding="utf-8")

df = df[["PERIODO", "VALOR"]].copy()

df.columns = ["Trimestre", "Variacion_PIB"]

df["Trimestre"] = df["Trimestre"].str.replace("T", "Q", regex=False)
df["Trimestre"] = pd.PeriodIndex(df["Trimestre"], freq="Q")

df = df.sort_values("Trimestre").reset_index(drop=True)

print(df.head())
print(df.tail())
