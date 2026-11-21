"""
VisionGuard AI - Production Inference Predictor Service
"""

import base64
import io
import os
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image

from src.inference.decision_engine import DecisionEngine
from src.models.efficientnet import MultiScaleFeatureExtractor
from src.models.feature_memory import PatchCoreMemoryBank
from src.utils.config import load_config


class VisionGuardPredictor:
    """Manages loaded PatchCore memory banks and DecisionEngines across categories."""

    def __init__(self, config_path: str | Path = "configs/model.yaml"):
        self.cfg = load_config(config_path)
        self.data_cfg = self.cfg["data"]
        self.pc_cfg = self.cfg["patchcore"]

        self.image_size = tuple(self.data_cfg["image_size"])  # (256, 256)
        self.artifact_root = Path(os.getenv("VISIONGUARD_MODEL_DIR", self.pc_cfg["artifact_root"]))
        self.categories: list[str] = list(self.data_cfg["categories"])

        # Shared frozen EfficientNetB0 backbone across all categories
        self.extractor = MultiScaleFeatureExtractor(
            input_shape=tuple(self.pc_cfg["input_shape"]),
            layer_names=self.pc_cfg["layer_names"],
            pool_size=self.pc_cfg["pool_size"],
        )

        self.banks: dict[str, PatchCoreMemoryBank] = {}
        self.engines: dict[str, DecisionEngine] = {}
        self.metadata: dict[str, dict[str, Any]] = {}

        self._load_all_categories()

    def _load_all_categories(self) -> None:
        for category in self.categories:
            cat_dir = self.artifact_root / category
            if not cat_dir.exists():
                raise FileNotFoundError(
                    f"Artifact directory missing for '{category}': {cat_dir}. "
                    "Run 'python -m scripts.build_artifacts' first."
                )
            bank, meta = PatchCoreMemoryBank.load(
                str(cat_dir), feature_extractor=self.extractor
            )
            engine = DecisionEngine.from_dict(meta["decision_engine"])
            self.banks[category] = bank
            self.engines[category] = engine
            self.metadata[category] = meta

    def _preprocess_bytes(self, image_bytes: bytes) -> np.ndarray:
        """Decodes raw bytes to a float32 RGB array of shape (H, W, 3) in [0, 1]."""
        try:
            pil_img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        except Exception as exc:
            raise ValueError(f"Invalid image file: {exc}") from exc

        pil_img = pil_img.resize((self.image_size[1], self.image_size[0]), Image.BILINEAR)
        arr = np.asarray(pil_img, dtype=np.float32) / 255.0
        return arr

    @staticmethod
    def _render_overlay_base64(
        img_rgb: np.ndarray,
        heatmap: np.ndarray,
        pass_threshold: float,
        fail_threshold: float,
        alpha: float = 0.45,
    ) -> str:
        """Renders a calibrated Jet heatmap overlay and encodes it as base64 PNG."""
        # Scale heatmap relative to calibrated thresholds so normal background stays cool blue
        denom = max(fail_threshold * 1.25 - pass_threshold * 0.5, 1e-6)
        norm_map = np.clip((heatmap - pass_threshold * 0.5) / denom, 0.0, 1.0)
        uint8_map = (norm_map * 255).astype(np.uint8)

        colored_bgr = cv2.applyColorMap(uint8_map, cv2.COLORMAP_JET)
        colored_rgb = cv2.cvtColor(colored_bgr, cv2.COLOR_BGR2RGB)

        base_uint8 = np.clip(img_rgb * 255.0, 0, 255).astype(np.uint8)
        blended = cv2.addWeighted(base_uint8, 1.0 - alpha, colored_rgb, alpha, 0)

        buf = io.BytesIO()
        Image.fromarray(blended).save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode("utf-8")

    def predict(self, image_bytes: bytes, category: str) -> dict[str, Any]:
        """Runs end-to-end anomaly scoring, 3-state routing, and heatmap overlay generation."""
        if category not in self.banks:
            raise KeyError(
                f"Unsupported category '{category}'. Available: {self.categories}"
            )

        t0 = time.perf_counter()
        img_rgb = self._preprocess_bytes(image_bytes)

        bank = self.banks[category]
        engine = self.engines[category]

        raw_score, heatmap = bank.score_image(
            img_rgb,
            gaussian_kernel=self.pc_cfg["gaussian_kernel"],
            sigma=self.pc_cfg["gaussian_sigma"],
        )
        decision_res = engine.evaluate(raw_score=raw_score, heatmap=heatmap)

        overlay_b64 = self._render_overlay_base64(
            img_rgb=img_rgb,
            heatmap=heatmap,
            pass_threshold=decision_res.pass_threshold,
            fail_threshold=decision_res.fail_threshold,
        )
        latency_ms = round((time.perf_counter() - t0) * 1000.0, 2)

        result = decision_res.to_dict()
        result["latency_ms"] = latency_ms
        result["overlay_base64"] = overlay_b64
        return result

