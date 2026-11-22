"""
VisionGuard AI - FastAPI Inference & Audit Service
"""

from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from pydantic import BaseModel

from src.inference.predictor import VisionGuardPredictor
from src.storage.history_db import InspectionHistoryDB


class InspectionResponse(BaseModel):
    inspection_id: int
    decision: str
    raw_score: float
    normalized_score: float
    z_score: float
    defect_area_pct: float
    pass_threshold: float
    fail_threshold: float
    category: str
    latency_ms: float
    overlay_base64: str


app_state: dict[str, Any] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    app_state["predictor"] = VisionGuardPredictor()
    app_state["db"] = InspectionHistoryDB()
    yield
    app_state.clear()


app = FastAPI(
    title="VisionGuard AI Inspection API",
    description="Real-time industrial surface anomaly detection powered by PatchCore.",
    version="1.0.0",
    lifespan=lifespan,
)


def get_predictor() -> VisionGuardPredictor:
    predictor = app_state.get("predictor")
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model artifacts not loaded.")
    return predictor


def get_db() -> InspectionHistoryDB:
    db = app_state.get("db")
    if db is None:
        raise HTTPException(status_code=503, detail="Inspection database unavailable.")
    return db


@app.get("/health")
def health_check() -> dict[str, Any]:
    predictor = get_predictor()
    return {
        "status": "healthy",
        "loaded_categories": predictor.categories,
    }


@app.get("/categories")
def list_categories() -> dict[str, Any]:
    predictor = get_predictor()
    details: list[dict[str, Any]] = []
    for cat in predictor.categories:
        eng = predictor.engines[cat]
        details.append(
            {
                "category": cat,
                "val_mean": round(eng.val_mean, 4),
                "val_std": round(eng.val_std, 4),
                "pass_threshold": round(eng.pass_threshold, 4),
                "fail_threshold": round(eng.fail_threshold, 4),
            }
        )
    return {"categories": details}


@app.post("/inspect", response_model=InspectionResponse)
async def inspect_part(
    category: str = Form(..., description="Product category: metal_nut, screw, or capsule"),
    file: UploadFile = File(..., description="Part image file (PNG or JPG)"),
) -> dict[str, Any]:
    predictor = get_predictor()
    db = get_db()

    if category not in predictor.banks:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown category '{category}'. Allowed: {predictor.categories}",
        )

    image_bytes = await file.read()
    if not image_bytes:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    try:
        result = predictor.predict(image_bytes=image_bytes, category=category)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    inspection_id = db.log_inspection(
        filename=file.filename or "uploaded_part.png",
        result=result,
    )
    result["inspection_id"] = inspection_id
    return result


@app.get("/history")
def inspection_history(limit: int = 50) -> dict[str, Any]:
    db = get_db()
    return {"records": db.get_recent(limit=limit)}


@app.get("/stats")
def inspection_stats() -> dict[str, Any]:
    db = get_db()
    return db.get_summary_stats()
