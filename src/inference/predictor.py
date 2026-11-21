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
        self.overrides_cfg = self.cfg.get("category_overrides", {})

        self.image_size = tuple(self.data_cfg["image_size"])
        self.artifact_root = Path(os.getenv("VISIONGUARD_MODEL_DIR", self.pc_cfg["artifact_root"]))
        self.categories: list[str] = list(self.data_cfg["categories"])

        self._extractors: dict[int, MultiScaleFeatureExtractor] = {}
        self.banks: dict[str, PatchCoreMemoryBank] = {}
        self.engines: dict[str, DecisionEngine] = {}
        self.metadata: dict[str, dict[str, Any]] = {}

        self._load_all_categories()

    def _get_extractor(self, pool_size: int) -> MultiScaleFeatureExtractor:
        if pool_size not in self._extractors:
            self._extractors[pool_size] = MultiScaleFeatureExtractor(
                input_shape=tuple(self.pc_cfg["input_shape"]),
                layer_names=self.pc_cfg["layer_names"],
                pool_size=pool_size,
            )
        return self._extractors[pool_size]

    def _load_all_categories(self) -> None:
        for category in self.categories:
            cat_dir = self.artifact_root / category
            if not cat_dir.exists():
                raise FileNotFoundError(
                    f"Artifact directory missing for '{category}': {cat_dir}. "
                    "Run 'python -m scripts.build_artifacts' first."
                )
            cat_ovr = self.overrides_cfg.get(category, {})
            pool_size = int(cat_ovr.get("pool_size", self.pc_cfg["pool_size"]))
            bank, meta = PatchCoreMemoryBank.load(
                str(cat_dir), feature_extractor=self._get_extractor(pool_size)
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
        max_alpha: float = 0.65,
    ) -> str:
        """Renders a localized anomaly heatmap overlay with transparent normal background."""
        floor = pass_threshold * 0.90
        ceil = max(fail_threshold * 1.15, float(np.max(heatmap)))
        denom = max(ceil - floor, 1e-6)
        norm_map = np.clip((heatmap - floor) / denom, 0.0, 1.0)

        # Suppress minor noise below 15% of active anomaly range so normal background stays crisp
        norm_map = np.where(norm_map < 0.15, 0.0, norm_map)
        uint8_map = (norm_map * 255).astype(np.uint8)

        colored_bgr = cv2.applyColorMap(uint8_map, cv2.COLORMAP_JET)
        colored_rgb = cv2.cvtColor(colored_bgr, cv2.COLOR_BGR2RGB).astype(np.float32)

        base_float = np.clip(img_rgb * 255.0, 0, 255)
        alpha_mask = (norm_map * max_alpha)[..., None]
        blended = (base_float * (1.0 - alpha_mask) + colored_rgb * alpha_mask).astype(np.uint8)

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
