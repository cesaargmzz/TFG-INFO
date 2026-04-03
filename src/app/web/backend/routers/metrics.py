from fastapi import APIRouter, HTTPException
from utils import load_metrics

router = APIRouter()

@router.get("/api/metrics/{bloque}/{geo}")
def get_metrics(bloque: int, geo: str):
    geo = geo.upper()
    if geo not in ("ES", "FR", "IT"):
        raise HTTPException(status_code=400, detail=f"País no válido: {geo}")
    if bloque not in (0, 1, 2, 3):
        raise HTTPException(status_code=400, detail=f"Bloque no válido: {bloque}")

    df = load_metrics(bloque, geo)
    if df.empty:
        raise HTTPException(status_code=404, detail="Datos no encontrados")

    return df.to_dict(orient="records")
