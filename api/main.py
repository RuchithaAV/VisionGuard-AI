"""
VisionGuard AI - FastAPI Inference Service
"""

from contextlib import asynccontextmanager
from typing import Any, Dict, List

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from pydantic import BaseModel

from src.inference.predictor import VisionGuardPredictor


class InspectionResponse(BaseModel):
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


predictor_state: Dict[str, VisionGuardPredictor] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    predictor_state["predictor"] = VisionGuardPredictor()
    yield
    predictor_state.clear()


app = FastAPI(
    title="VisionGuard AI Inspection API",
    description="Real-time industrial surface anomaly detection powered by PatchCore.",
    version="1.0.0",
    lifespan=lifespan,
)


def get_predictor() -> VisionGuardPredictor:
    predictor = predictor_state.get("predictor")
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model artifacts not loaded.")
    return predictor


@app.get("/health")
def health_check() -> Dict[str, Any]:
    predictor = get_predictor()
    return {
        "status": "healthy",
        "loaded_categories": predictor.categories,
    }


@app.get("/categories")
def list_categories() -> Dict[str, Any]:
    predictor = get_predictor()
    details: List[Dict[str, Any]] = []
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
) -> Dict[str, Any]:
    predictor = get_predictor()
    if category not in predictor.banks:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown category '{category}'. Allowed: {predictor.categories}",
        )

    image_bytes = await file.read()
    if not image_bytes:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    try:
        return predictor.predict(image_bytes=image_bytes, category=category)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
