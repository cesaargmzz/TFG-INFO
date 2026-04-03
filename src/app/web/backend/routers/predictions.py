from fastapi import APIRouter, HTTPException
from utils import load_predictions

router = APIRouter()

@router.get("/api/predictions/{bloque}/{geo}/{modelo}")
def get_predictions(bloque: int, geo: str, modelo: str):
    geo = geo.upper()
    if geo not in ("ES", "FR", "IT"):
        raise HTTPException(status_code=400, detail=f"País no válido: {geo}")
    if bloque not in (0, 1, 2, 3):
        raise HTTPException(status_code=400, detail=f"Bloque no válido: {bloque}")
    if modelo not in ("base", "ext"):
        raise HTTPException(status_code=400, detail=f"Modelo no válido: {modelo}")
    if bloque == 3 and modelo == "base":
        raise HTTPException(status_code=400, detail="Bloque 3 solo tiene modelo ext")

    df = load_predictions(bloque, geo, modelo)
    if df.empty:
        raise HTTPException(status_code=404, detail="Datos no encontrados")

    return df.to_dict(orient="records")
