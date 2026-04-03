from fastapi import APIRouter, HTTPException
from utils import load_all_metrics
import math

router = APIRouter()

def sanitize(val):
    if isinstance(val, float) and math.isnan(val):
        return None
    return val

@router.get("/api/summary")
def get_summary():
    df = load_all_metrics()
    if df.empty:
        raise HTTPException(status_code=404, detail="No hay datos de métricas")
    records = df.to_dict(orient="records")
    return [{k: sanitize(v) for k, v in row.items()} for row in records]
