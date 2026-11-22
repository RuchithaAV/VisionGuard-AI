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
from src.inference.localization import extract_foreground_mask
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

        self.default_image_size = tuple(self.data_cfg["image_size"])
        self.artifact_root = Path(os.getenv("VISIONGUARD_MODEL_DIR", self.pc_cfg["artifact_root"]))
        self.categories: list[str] = list(self.data_cfg["categories"])

        self._extractors: dict[tuple, MultiScaleFeatureExtractor] = {}
        self.banks: dict[str, PatchCoreMemoryBank] = {}
        self.engines: dict[str, DecisionEngine] = {}
        self.metadata: dict[str, dict[str, Any]] = {}
        self.category_image_sizes: dict[str, tuple[int, int]] = {}

        self._load_all_categories()

    def _get_extractor(
        self, input_shape: tuple[int, int, int], layer_names: tuple[str, ...], pool_size: int
    ) -> MultiScaleFeatureExtractor:
        key = (input_shape, layer_names, pool_size)
        if key not in self._extractors:
            self._extractors[key] = MultiScaleFeatureExtractor(
                input_shape=input_shape,
                layer_names=list(layer_names),
                pool_size=pool_size,
            )
        return self._extractors[key]

    def _load_all_categories(self) -> None:
        for category in self.categories:
            cat_dir = self.artifact_root / category
            if not cat_dir.exists():
                raise FileNotFoundError(
                    f"Artifact directory missing for '{category}': {cat_dir}. "
                    "Run 'python -m scripts.build_artifacts' first."
                )
            cat_ovr = self.overrides_cfg.get(category, {})
            img_size = tuple(cat_ovr.get("image_size", self.default_image_size))
            in_shape = tuple(cat_ovr.get("input_shape", self.pc_cfg["input_shape"]))
            layers = tuple(cat_ovr.get("layer_names", self.pc_cfg["layer_names"]))
            pool_size = int(cat_ovr.get("pool_size", self.pc_cfg["pool_size"]))

            self.category_image_sizes[category] = img_size
            extractor = self._get_extractor(in_shape, layers, pool_size)
            bank, meta = PatchCoreMemoryBank.load(str(cat_dir), feature_extractor=extractor)
            engine = DecisionEngine.from_dict(meta["decision_engine"])
            self.banks[category] = bank
            self.engines[category] = engine
            self.metadata[category] = meta

    def _preprocess_bytes(self, image_bytes: bytes, target_size: tuple[int, int]) -> np.ndarray:
        """Decodes raw bytes to a float32 RGB array of shape (H, W, 3) in [0, 1]."""
        try:
            pil_img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        except Exception as exc:
            raise ValueError(f"Invalid image file: {exc}") from exc

        pil_img = pil_img.resize((target_size[1], target_size[0]), Image.BILINEAR)
        return np.asarray(pil_img, dtype=np.float32) / 255.0

    @staticmethod
    def _render_overlay_base64(
        img_rgb: np.ndarray,
        heatmap: np.ndarray,
        pass_threshold: float,
        fail_threshold: float,
        decision: str,
        category: str = "unknown",
        max_alpha: float = 0.70,
    ) -> str:
        """Highlights the anomaly hotspot with foreground-relative thresholds."""
        base_uint8 = np.clip(img_rgb * 255.0, 0, 255).astype(np.uint8)
        peak = float(np.max(heatmap))
        peak = max(peak, 1e-6)

        # Segment the part before rendering. This suppresses blank-background
        # hotspots and keeps the heatmap anchored to the screw/metal surface.
        erode_pixels = 4 if category == "screw" else 6
        foreground = extract_foreground_mask(
            img_rgb,
            erode_pixels=erode_pixels,
            border_patch=16,
        )

        fg_values = heatmap[foreground > 0]
        if fg_values.size:
            active_floor = float(np.percentile(fg_values, 90.0))
            active_ceiling = float(np.percentile(fg_values, 99.5))
        else:
            active_floor = float(np.percentile(heatmap, 90.0))
            active_ceiling = float(np.percentile(heatmap, 99.5))

        # Contrast stretching from the 90th to 99.5th percentile exposes small
        # screw defects while avoiding an almost-uniform red/blue diagonal band.
        floor = min(active_floor, peak * 0.75)
        ceiling = max(active_ceiling, peak)
        denom = max(ceiling - floor, 1e-6)
        norm_map = np.clip((heatmap - floor) / denom, 0.0, 1.0)
        norm_map = np.power(norm_map, 0.75)

        # Keep a weak JET heatmap for every inspection. For normal/PASS images,
        # the low-confidence pixels remain subtle rather than disappearing.
        if decision == "PASS":
            norm_map *= 0.35
        norm_map *= foreground

        uint8_map = (norm_map * 255).astype(np.uint8)
        colored_bgr = cv2.applyColorMap(uint8_map, cv2.COLORMAP_JET)
        colored_rgb = cv2.cvtColor(colored_bgr, cv2.COLOR_BGR2RGB).astype(np.float32)

        alpha_limit = max_alpha if decision != "PASS" else 0.30
        alpha_mask = (norm_map * alpha_limit)[..., None]
        blended = (base_uint8.astype(np.float32) * (1.0 - alpha_mask) + colored_rgb * alpha_mask).astype(np.uint8)

        # Draw a crisp contour around the strongest connected hotspot.
        core_ratio = 0.72 if category == "screw" else 0.80
        defect_threshold = max(peak * core_ratio, ceiling * 0.62)
        defect_bin = ((heatmap >= defect_threshold) & (foreground > 0)).astype(np.uint8) * 255
        close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        open_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        defect_bin = cv2.morphologyEx(defect_bin, cv2.MORPH_CLOSE, close_kernel)
        defect_bin = cv2.morphologyEx(defect_bin, cv2.MORPH_OPEN, open_kernel)
        contours, _ = cv2.findContours(defect_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        min_contour_area = (base_uint8.shape[0] * base_uint8.shape[1]) * (0.0001 if category == "screw" else 0.0005)
        for contour in contours:
            if cv2.contourArea(contour) >= min_contour_area:
                cv2.drawContours(blended, [contour], -1, (255, 45, 45), thickness=2)

        buf = io.BytesIO()
        Image.fromarray(blended).save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode("utf-8")

    @staticmethod
    def _find_category_mismatch(
        category_z_scores: dict[str, float],
        selected_category: str,
    ) -> str | None:
        """Returns another category only when it fits clearly better than selection."""
        if selected_category not in category_z_scores:
            return None

        selected_z = category_z_scores[selected_category]
        if selected_z < 2.0:
            return None

        best_category = min(category_z_scores, key=category_z_scores.get)
        best_z = category_z_scores[best_category]
        if (
            best_category != selected_category
            and best_z <= 2.0
            and selected_z - best_z >= 3.0
        ):
            return best_category
        return None

    def _score_category_z_scores(
        self,
        image_bytes: bytes,
    ) -> dict[str, float]:
        scores: dict[str, float] = {}
        for category in self.categories:
            target_size = self.category_image_sizes[category]
            img_rgb = self._preprocess_bytes(image_bytes, target_size=target_size)
            raw_score, _ = self.banks[category].score_image(
                img_rgb,
                gaussian_kernel=self.pc_cfg["gaussian_kernel"],
                sigma=self.pc_cfg["gaussian_sigma"],
            )
            engine = self.engines[category]
            scores[category] = (raw_score - engine.val_mean) / engine.val_std
        return scores

    def predict(self, image_bytes: bytes, category: str) -> dict[str, Any]:
        """Runs end-to-end anomaly scoring, 3-state routing, and heatmap overlay generation."""
        if category not in self.banks:
            raise KeyError(
                f"Unsupported category '{category}'. Available: {self.categories}"
            )

        t0 = time.perf_counter()
        target_size = self.category_image_sizes[category]
        img_rgb = self._preprocess_bytes(image_bytes, target_size=target_size)

        bank = self.banks[category]
        engine = self.engines[category]

        raw_score, heatmap = bank.score_image(
            img_rgb,
            gaussian_kernel=self.pc_cfg["gaussian_kernel"],
            sigma=self.pc_cfg["gaussian_sigma"],
        )
        decision_res = engine.evaluate(raw_score=raw_score, heatmap=heatmap)

        if decision_res.decision != "PASS":
            category_z_scores = self._score_category_z_scores(image_bytes)
            category_z_scores[category] = decision_res.z_score
            matched_category = self._find_category_mismatch(
                category_z_scores,
                category,
            )
            if matched_category is not None:
                raise ValueError(
                    f"Category mismatch: this appears to be a "
                    f"{matched_category.replace('_', ' ')} object, but "
                    f"{category.replace('_', ' ')} was selected."
                )

        overlay_b64 = self._render_overlay_base64(
            img_rgb=img_rgb,
            heatmap=heatmap,
            pass_threshold=decision_res.pass_threshold,
            fail_threshold=decision_res.fail_threshold,
            decision=decision_res.decision,
            category=category,
        )
        latency_ms = round((time.perf_counter() - t0) * 1000.0, 2)

        result = decision_res.to_dict()
        result["latency_ms"] = latency_ms
        result["overlay_base64"] = overlay_b64
        return result
