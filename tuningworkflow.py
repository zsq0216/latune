import time
from typing import List, Dict, Tuple, Any, Optional
import numpy as np
import json
import os
from pathlib import Path
import matplotlib.pyplot as plt

from latune import LaTune
from llama_executor import LlamaExecutor
from hv_calculator import HypervolumeCalculator


class TuningWorkflow:
    """
    A general tuning workflow that encapsulates the common logic of scripts 02 and 03,
    """

    def __init__(
        self,
        parameters_path: str,
        objectives: Dict[str, str],
        max_observations: int = 30,
        parallel_degree: int = 5,
        device: str = "gpu",
        hardware: str = "rtx3060",
        model: str = "qwen3-4b",
        quant: str = "q4",
        *,
        model_dir: str = "./../models",           # where gguf lives
        bounds_dir: str = "bounds",               # where metric bounds live
        surrogate_dir: str = "surrogate_models",  # where surrogate saved
        pareto_dir: str = "pareto_fronts",        # where pareto saved
        hv_dir: str = "hv_progress",              # where hv progress saved
    ):
        # core
        self.tuner = LaTune(parameters_path, objectives)
        self.objectives = objectives
        self.max_observations = max_observations
        self.parallel_degree = parallel_degree
        self.executor = LlamaExecutor(self.tuner.param_types, device=device, hardware=hardware)

        # id / paths
        self.hardware = hardware
        self.model_name = f"{model}-{quant}"
        self.model = f"{model_dir}/{self.model_name}.gguf"

        self.bounds_dir = bounds_dir
        self.surrogate_dir = surrogate_dir
        self.pareto_dir = pareto_dir
        self.hv_dir = hv_dir

        # state
        self.current_observations = 0
        self.history = {"configs": [], "performance": []}
        self.iter_hv: List[float] = []

        # HV
        bounds_path = f"{self.bounds_dir}/{self.hardware}/{self.model_name}.json"
        self.bounds = self._load_metric_bounds(bounds_path)
        self.hv_calc = HypervolumeCalculator(self.bounds)

        # iteration index (mirror LaTune.iteration behavior if needed)
        if not hasattr(self.tuner, "iteration"):
            self.tuner.iteration = 0

    # ---------- public API ----------

    def run_workflow(self):
        """Run full loop: init, iterate suggest/eval, update, stop by budget."""
        # 1) init design+eval
        initial_samples = self._get_initial_points()
        self._evaluate_configs(initial_samples, init_model=True)

        # 2) optimization loop
        while self.current_observations < self.max_observations:
            start_time = time.time()

            suggested_configs = self.tuner.suggest_configurations(k=self.parallel_degree)
            new_perfs = self._evaluate_configs(suggested_configs)

            # update model
            self.tuner.update_surrogate(len(new_perfs))
            self.tuner.update_pareto_front()
            self.tuner.set_reference_point()

            iteration_time = time.time() - start_time
            print(
                f"Iteration {self.tuner.iteration} completed. "
                f"Observations: {self.current_observations}/{self.max_observations} "
                f"Time: {iteration_time:.1f}s"
            )
            self.tuner.iteration += 1

            hv = self._compute_hypervolume()
            self.iter_hv.append(hv)

            if self.current_observations >= self.max_observations:
                # save surrogate
                out_path = f"{self.surrogate_dir}/{self.hardware}/{self.model_name}.pth"
                Path(out_path).parent.mkdir(parents=True, exist_ok=True)
                self.tuner.save_surrogate(out_path)
                # self._plot_hv_over_iterations()
                break

    def get_best_config(self) -> Dict:
        """
        Extract best config from history or pareto front
        """
        if len(self.tuner.objectives) == 1:
            return self._get_single_best()
        else:
            return self._get_pareto_best()

    def save_pareto_front_and_hv(self, model_tag: Optional[str] = None):
        """
        Save pareto front and hv progress to disk:
          pareto_fronts/{hardware}/{model_tag}-latune.json
          hv_progress/{hardware}/{model_tag}-latune.json
        """
        if len(self.objectives) == 1:
            print("Saving Pareto front is only supported for multi-objective runs.")
            return

        if not hasattr(self.tuner, "pareto_front"):
            print("No Pareto front is available to save.")
            return

        model_tag = model_tag or self.model_name
        pareto_serializable = [
            {"config": self._convert_to_serializable(cfg), "perf": self._convert_to_serializable(perf)}
            for cfg, perf in self.tuner.pareto_front
        ]

        pareto_path = f"{self.pareto_dir}/{self.hardware}/{model_tag}-latune.json"
        hv_path = f"{self.hv_dir}/{self.hardware}/{model_tag}-latune.json"
        Path(pareto_path).parent.mkdir(parents=True, exist_ok=True)
        Path(hv_path).parent.mkdir(parents=True, exist_ok=True)

        with open(pareto_path, "w", encoding="utf-8") as f:
            json.dump(pareto_serializable, f, indent=2)
        print(f"Pareto front saved to {pareto_path}")

        with open(hv_path, "w", encoding="utf-8") as f:
            json.dump(self.iter_hv, f, indent=4)
        print(f"Hypervolume progress saved to {hv_path}")

    def load_pareto_front(self, filepath: str):
        """
        Load pareto front from a JSON file.
        """
        p = Path(filepath)
        if not p.exists():
            print(f"File {filepath} does not exist.")
            return

        with open(p, "r", encoding="utf-8") as f:
            pareto_data = json.load(f)
        self.tuner.pareto_front = [(item["config"], item["perf"]) for item in pareto_data]
        print(f"Pareto front loaded from {filepath}.")

    def visualize_pareto_front(self):
        """
        Visualize the current Pareto front (2-objective case only).
        """
        front = self.tuner.get_pareto_front()
        perfs = [sample[1] for sample in front]
        if len(self.tuner.objectives) != 2:
            print("Visualization currently supports the 2-objective case only.")
            return

        y = np.array([perf["tps_avg"] for perf in perfs])
        x = np.array([perf["gpu_p95"] for perf in perfs])

        plt.figure()
        plt.scatter(x, y, c="red", label="Pareto Front")
        plt.xlabel("gpu_p95")
        plt.ylabel("tps_avg")
        plt.title("Pareto Front Evolution")
        plt.legend()
        plt.tight_layout()
        plt.savefig("tps_gpu_r3.pdf")
        plt.show()

    # ---------- internals ----------

    def _get_initial_points(self) -> List[Dict]:
        history_path = self._get_history_path(self.model, self.hardware)
        if history_path is not None:
            print(f"[Init] Warm-start from history: {history_path}")
            return self.tuner.load_configs_from_history(history_path)
        # default: generate
        print("[Init] Generate initial samples via LaTune.")
        return self.tuner.generate_initial_samples(5)
    
    def _get_history_path(self, model_name, hardware, filepath="meta_features/records.jsonl"):
        """Find the best historical record matching (model_name, hardware)."""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Records file not found: {filepath}")

        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue

                if rec.get("model_name") == model_name and rec.get("hardware") == hardware:
                    top = rec.get("top_record", {})
                    print(f"find top")
                    {
                        "top_model": top.get("model_name"),
                        "top_hardware": top.get("hardware"),
                        "similarity": top.get("similarity"),
                    }
                    return f"pareto_fronts/{top.get('hardware')}/{top.get('model_name')}-latune.json"

        return None

    def _load_metric_bounds(self, path: str) -> Dict[str, Tuple[float, float]]:
        """
        Load metric bounds from a JSON file.
        """
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        out: Dict[str, Tuple[float, float]] = {}
        for metric, mm in data.items():
            if not isinstance(mm, dict):
                continue
            lo = mm.get("min")
            hi = mm.get("max")
            if lo is None or hi is None:
                continue
            try:
                lo = float(lo); hi = float(hi)
            except (TypeError, ValueError):
                continue
            out[metric] = (lo, hi)
        return out

    def _evaluate_configs(self, configs: List[Dict], init_model=False) -> List[Dict]:
        """
        Evaluate a batch of configurations, update history, and optionally initialize the surrogate model.
        """
        perf_results: List[Dict] = []
        for config in configs:
            try:
                result = self.executor.run_server_performance_test(config, model_path=self.model)
                perf = {m: result[m] for m in self.objectives}
            except ValueError:
                continue

            self.tuner.observations.append((config, perf))
            print(f"config: {config}, perf: {perf}")

            self.history["configs"].append(config)
            self.history["performance"].append(perf)
            self.current_observations += 1
            perf_results.append(perf)

            if self.current_observations >= self.max_observations:
                break

        if init_model:
            self.tuner.update_surrogate()
            self.tuner.update_pareto_front()
            self.tuner.set_reference_point()

            hv = self._compute_hypervolume()
            self.iter_hv.append(hv)
            print("== SURROGATE MODEL UPDATED ==")

        return perf_results

    def _get_single_best(self) -> Dict:
        obj = list(self.objectives.keys())[0]
        direct = list(self.objectives.values())[0]
        perfs = [perf[obj] for perf in self.history["performance"]]
        if not perfs:
            raise RuntimeError("No performance data available.")
        best_idx = int(np.argmin(perfs) if direct == "min" else np.argmax(perfs))
        print(self.history["performance"][best_idx])
        return self.history["configs"][best_idx]

    def _get_pareto_best(self) -> Dict:
        pareto_front = self.tuner.get_pareto_front()
        print(pareto_front)
        perfs = [sample[1] for sample in pareto_front]
        best_idx, best_score = self.tuner.evaluate_pareto(perfs)
        print(best_score)
        print(pareto_front[best_idx])
        return pareto_front[best_idx][0]

    def _convert_to_serializable(self, obj: Any) -> Any:
        if isinstance(obj, (np.integer,)):
            return int(obj)
        elif isinstance(obj, (np.floating,)):
            return float(obj)
        elif isinstance(obj, (np.bool_,)):
            return bool(obj)
        elif isinstance(obj, dict):
            return {k: self._convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_to_serializable(i) for i in obj]
        else:
            return obj

    def _compute_hypervolume(self) -> float:
        if not self.history["performance"]:
            return 0.0
        pareto_front = [t[1] for t in self.tuner.get_pareto_front()]
        if not pareto_front:
            return 0.0
        hv = self.hv_calc.compute(pareto_front)
        print(f"Current Hypervolume: {hv:.4f}")
        return hv

    def _plot_hv_over_iterations(self):
        plt.figure(figsize=(8, 5))
        plt.plot(range(1, len(self.iter_hv) + 1), self.iter_hv, marker="o")
        plt.xlabel("Iteration")
        plt.ylabel("Hypervolume")
        plt.title("Best Hypervolume Over Iterations")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig("hypervolume_progress_latune.png")
        plt.show()
