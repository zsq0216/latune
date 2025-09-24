import numpy as np
from pymoo.indicators.hv import Hypervolume


class HypervolumeCalculator:
    def __init__(self,
                 bounds,
                 maximize=["tps_avg"],
                 minimize=["gpu_avg"]):
        """
        Args:
            bounds: 每个目标的固定上下界，例如
                {
                    "tps_avg": (50, 200),
                    "gpu_avg": (20, 100)
                }
            maximize: 要最大化的目标
            minimize: 要最小化的目标
        """
        self.bounds = bounds
        self.maximize = maximize
        self.minimize = minimize

    def _normalize(self, point):
        """将原始点归一化到 [0,1]"""
        norm = []
        for key in self.minimize:
            lo, hi = self.bounds[key]
            norm.append((point[key] - lo) / (hi - lo + 1e-12))

        for key in self.maximize:
            lo, hi = self.bounds[key]
            norm.append((hi - point[key]) / (hi - lo + 1e-12))

        return norm

    def compute(self, raw_points):
        if not raw_points:
            return 0.0

        # 归一化点集
        norm_points = np.array([self._normalize(p) for p in raw_points])

        # 参考点：全 1 的最差点
        ref_point = np.ones(norm_points.shape[1])

        hv = Hypervolume(ref_point=ref_point)
        return hv(norm_points)

