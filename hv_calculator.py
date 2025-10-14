import numpy as np
from pymoo.indicators.hv import Hypervolume


class HypervolumeCalculator:
    """
    Compute hypervolume for multi-objective optimization results.
    """

    def __init__(self, bounds, maximize=["tps_avg"], minimize=["gpu_p95"]):
        """
        Args:
            bounds (dict): e.g. {"tps_avg": (50, 200), "gpu_p95": (20, 100)}
            maximize (list): objectives to maximize
            minimize (list): objectives to minimize
        """
        self.bounds = bounds
        self.maximize = maximize
        self.minimize = minimize

    def _normalize(self, point):
        """Normalize a point to [0, 1] range."""
        norm = []
        # Minimize objectives
        for key in self.minimize:
            lo, hi = self.bounds[key]
            norm.append((point[key] - lo) / (hi - lo + 1e-12))
        # Maximize objectives
        for key in self.maximize:
            lo, hi = self.bounds[key]
            norm.append((hi - point[key]) / (hi - lo + 1e-12))
        return norm

    def compute(self, raw_points):
        """Return hypervolume value for given data points."""
        if not raw_points:
            return 0.0

        norm_points = np.array([self._normalize(p) for p in raw_points])
        ref_point = np.ones(norm_points.shape[1])
        hv = Hypervolume(ref_point=ref_point)
        return hv(norm_points)
