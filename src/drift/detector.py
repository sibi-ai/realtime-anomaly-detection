"""Concept drift detection using statistical tests on incoming data."""
import numpy as np
from scipy import stats
import structlog

log = structlog.get_logger()


class KSDriftDetector:
    """
    Kolmogorov-Smirnov drift detector.
    Compares current window distribution against reference (training) distribution.
    """

    def __init__(self, reference: np.ndarray, threshold: float = 0.05,
                 window_size: int = 1000):
        self.reference  = reference
        self.threshold  = threshold
        self.window_size = window_size
        self._window: list[float] = []

    def update(self, value: float) -> bool:
        """Add new observation. Returns True if drift detected."""
        self._window.append(value)
        if len(self._window) > self.window_size:
            self._window.pop(0)
        if len(self._window) < self.window_size:
            return False
        return self._check()

    def _check(self) -> bool:
        stat, p_value = stats.ks_2samp(self.reference, np.array(self._window))
        drift = p_value < self.threshold
        if drift:
            log.warning("drift_detected", ks_stat=round(stat, 4), p_value=round(p_value, 6))
        return drift

    def reset(self) -> None:
        self._window.clear()
