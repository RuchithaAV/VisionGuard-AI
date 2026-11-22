"""
VisionGuard AI - Calibrated 3-State Decision Engine Unit Tests
"""

import numpy as np
import pytest

from src.inference.decision_engine import DecisionEngine


def test_decision_engine_calibration_and_three_states():
    """Verify PASS, REVIEW, and FAIL boundaries calibrated from normal validation scores."""
    # Normal validation distribution with mean ~18.0, std ~1.0
    val_scores = np.array([17.0, 17.5, 18.0, 18.5, 19.0])
    engine = DecisionEngine(pass_std_multiplier=2.0, fail_std_multiplier=3.5, category="metal_nut")
    engine.fit(val_scores)

    assert engine.is_calibrated
    assert engine.pass_threshold < engine.fail_threshold

    # 1. Nominal part -> PASS
    res_pass = engine.evaluate(raw_score=engine.val_mean)
    assert res_pass.decision == "PASS"
    assert res_pass.normalized_score == pytest.approx(0.25)

    # 2. Borderline part -> REVIEW
    mid_review = 0.5 * (engine.pass_threshold + engine.fail_threshold)
    res_review = engine.evaluate(raw_score=mid_review)
    assert res_review.decision == "REVIEW"
    assert 0.50 < res_review.normalized_score < 0.80

    # 3. Clear defect -> FAIL
    res_fail = engine.evaluate(raw_score=engine.fail_threshold + 2.0)
    assert res_fail.decision == "FAIL"
    assert res_fail.normalized_score > 0.80


def test_defect_area_percentage_calculation():
    """Verify defect footprint percentage counts pixels exceeding pass_threshold."""
    engine = DecisionEngine(pass_std_multiplier=2.0, fail_std_multiplier=3.5)
    engine.fit(np.array([10.0, 10.0, 10.0]))  # val_mean=10.0, val_std=1e-6

    # 10x10 heatmap where 25 of 100 pixels (25.0%) exceed pass_threshold
    heatmap = np.full((10, 10), 5.0, dtype=np.float32)
    heatmap[:5, :5] = 25.0

    result = engine.evaluate(raw_score=25.0, heatmap=heatmap)
    assert result.decision == "FAIL"
    assert result.defect_area_pct == pytest.approx(25.0)


def test_defect_area_uses_peak_relative_threshold():
    engine = DecisionEngine(
        pass_std_multiplier=2.0,
        fail_std_multiplier=3.5,
        val_mean=10.0,
        val_std=1.0,
    )
    heatmap = np.full((100, 100), 5.0, dtype=np.float32)
    heatmap[40:50, 40:50] = 20.0

    result = engine.evaluate(raw_score=20.0, heatmap=heatmap)

    assert result.defect_area_pct == pytest.approx(1.0)


def test_decision_engine_serialization_roundtrip():
    """Verify to_dict() and from_dict() preserve calibrated thresholds exactly."""
    engine = DecisionEngine(pass_std_multiplier=2.0, fail_std_multiplier=3.5, category="capsule")
    engine.fit(np.array([11.5, 12.0, 12.5, 13.0]))

    payload = engine.to_dict()
    restored = DecisionEngine.from_dict(payload)

    assert restored.category == "capsule"
    assert restored.pass_threshold == pytest.approx(engine.pass_threshold)
    assert restored.fail_threshold == pytest.approx(engine.fail_threshold)
    assert restored.evaluate(16.0).decision == engine.evaluate(16.0).decision


def test_uncalibrated_or_invalid_multipliers_raise():
    """Verify guardrails against uncalibrated calls and invalid multiplier ordering."""
    with pytest.raises(ValueError):
        DecisionEngine(pass_std_multiplier=3.5, fail_std_multiplier=2.0)

    uncalibrated = DecisionEngine()
    with pytest.raises(ValueError):
        uncalibrated.evaluate(15.0)