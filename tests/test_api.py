"""
VisionGuard AI - Automated Unit & API Integration Tests
"""

import io

import numpy as np
from fastapi.testclient import TestClient
from PIL import Image

from api.main import app
from src.inference.decision_engine import DecisionEngine
from src.utils.config import load_config


def test_config_loads_valid_structure():
    cfg = load_config("configs/model.yaml")
    assert "data" in cfg
    assert "patchcore" in cfg
    assert set(cfg["data"]["categories"]) == {"metal_nut", "screw", "capsule"}


def test_decision_engine_routing():
    engine = DecisionEngine(
        category="metal_nut",
        val_mean=18.0,
        val_std=1.5,
        pass_std_multiplier=2.0,
        fail_std_multiplier=3.5,
    )
    flat_map = np.zeros((256, 256), dtype=np.float32)

    assert engine.evaluate(raw_score=19.5, heatmap=flat_map).decision == "PASS"
    assert engine.evaluate(raw_score=22.0, heatmap=flat_map).decision == "REVIEW"
    assert engine.evaluate(raw_score=25.0, heatmap=flat_map).decision == "FAIL"


def test_api_endpoints_and_inspection(tmp_path, monkeypatch):
    monkeypatch.setenv("INSPECTION_DB_PATH", str(tmp_path / "test_inspections.db"))

    buf = io.BytesIO()
    Image.fromarray(np.zeros((256, 256, 3), dtype=np.uint8)).save(buf, format="PNG")
    img_bytes = buf.getvalue()

    with TestClient(app) as client:
        health = client.get("/health")
        assert health.status_code == 200
        assert health.json()["status"] == "healthy"

        cats = client.get("/categories")
        assert cats.status_code == 200
        assert len(cats.json()["categories"]) == 3

        resp = client.post(
            "/inspect",
            data={"category": "metal_nut"},
            files={"file": ("synthetic.png", img_bytes, "image/png")},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["decision"] in {"PASS", "REVIEW", "FAIL"}
        assert "overlay_base64" in body and len(body["overlay_base64"]) > 100

        stats = client.get("/stats").json()
        assert stats["total_inspected"] >= 1
