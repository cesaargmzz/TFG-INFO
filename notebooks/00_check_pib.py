import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]  # sube de notebooks/ a la raíz
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))


from src.data.loaders import load_pib, save_parquet


def main():
    # Cargar datos del PIB desde el CSV
    df = load_pib(Path("data/PIB.csv"))

    # Mostrar las 5 primeras y 5 últimas filas
    print("\nPrimeras filas:")
    print(df.head())
    print("\nÚltimas filas:")
    print(df.tail())

    # Guardar el DataFrame limpio en formato Parquet
    output_path = save_parquet(df, "pib_trimestral")
    print(f"\nArchivo guardado en: {output_path}")


if __name__ == "__main__":
    main()
