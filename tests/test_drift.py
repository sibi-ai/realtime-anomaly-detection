"""Tests for KS drift detector."""
import numpy as np
import pytest
from src.drift.detector import KSDriftDetector


def test_no_drift_same_distribution():
    ref = np.random.default_rng(42).normal(0, 1, 2000)
    det = KSDriftDetector(reference=ref, threshold=0.01, window_size=500)
    same_dist = np.random.default_rng(7).normal(0, 1, 500)
    drift_detected = any(det.update(v) for v in same_dist)
    assert not drift_detected


def test_drift_different_distribution():
    ref = np.random.default_rng(42).normal(0, 1, 2000)
    det = KSDriftDetector(reference=ref, threshold=0.05, window_size=500)
    # Shift mean by 5 sigma — obvious drift
    shifted = np.random.default_rng(7).normal(5, 1, 500)
    results = [det.update(v) for v in shifted]
    assert any(results), "Drift should be detected for large distribution shift"


def test_reset_clears_window():
    ref = np.ones(2000)
    det = KSDriftDetector(reference=ref, window_size=100)
    for _ in range(50):
        det.update(1.0)
    det.reset()
    assert len(det._window) == 0
