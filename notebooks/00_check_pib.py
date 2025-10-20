# notebooks/00_check_pib.py

import sys
from pathlib import Path

# --- 1) Añadir la raíz del proyecto al sys.path ---
ROOT = Path(__file__).resolve().parents[1]  # sube de notebooks/ a la raíz
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

# --- 2) Importar el loader ya hecho ---
from src.data.loaders import load_pib, save_parquet


def main():
    # (opcional) comprobación amistosa de que existe el CSV en la raíz/data
    expected_csv = ROOT / "data" / "PIB.csv"
    if not expected_csv.exists():
        raise FileNotFoundError(
            f"No encuentro el CSV en {expected_csv}\n"
            "Asegúrate de que el fichero se llama exactamente 'PIB.csv' "
            "y está dentro de la carpeta 'data' en la raíz del proyecto."
        )

    # 3) Importante: NO pasar ruta -> el loader resuelve bien la ruta
    df = load_pib()  # usa data/PIB.csv relativo a la raíz del proyecto

    print("\nPrimeras filas:")
    print(df.head())

    print("\nÚltimas filas:")
    print(df.tail())

    output_path = save_parquet(df, "pib_trimestral")
    print(f"\nArchivo guardado en: {output_path}")


if __name__ == "__main__":
    main()
