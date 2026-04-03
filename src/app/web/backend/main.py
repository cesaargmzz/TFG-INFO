from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from routers import predictions, metrics, summary

app = FastAPI(title="GDPulse API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_methods=["GET"],
    allow_headers=["*"],
)

app.include_router(predictions.router)
app.include_router(metrics.router)
app.include_router(summary.router)

@app.get("/api/health")
def health():
    return {"status": "ok"}
