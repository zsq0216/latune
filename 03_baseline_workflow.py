from hv_calculator import HypervolumeCalculator
import json
import argparse
import matplotlib.pyplot as plt
from typing import List, Dict, Any, Tuple, Optional
import numpy as np
from basetuner import DefaultTuner, RandomTuner, GeneticAlgorithmTuner, ConstrainedBayesTuner, ScootTuner


class BaselineWorkflow:
    """
    Generic tuning workflow wrapper that:
      - Instantiates a tuner by algorithm key
      - Runs iterative suggest -> evaluate -> update loop
      - Tracks Pareto front and hypervolume across iterations
      - Saves Pareto front and HV progress to disk
    """

    def __init__(self,
                 parameters_path: str,
                 known_constraints: List[str],
                 objectives: Dict[str, str],
                 algorithm: str = "RD",
                 max_observations: int = 30,
                 parallel_degree: int = 5,
                 device: str = "gpu",
                 hardware: str = "m4",
                 model: str = "qwen3-8b",
                 quant: str = "q4"):

        # Select tuner by algorithm key
        self.method = algorithm
        if algorithm == "Default":
            self.tuner = DefaultTuner(parameters_path, known_constraints, objectives, device=device, hardware=hardware)
        elif algorithm == "RD":
            self.tuner = RandomTuner(parameters_path, known_constraints, objectives, device=device, hardware=hardware)
        elif algorithm == "GA":
            self.tuner = GeneticAlgorithmTuner(parameters_path, known_constraints, objectives, device=device, hardware=hardware)
        elif algorithm == "CBO":
            self.tuner = ConstrainedBayesTuner(parameters_path, known_constraints, objectives,
                                               device=device, hardware=hardware, model_name=f"{model}-{quant}")
        elif algorithm == "SCOOT":
            self.tuner = ScootTuner(parameters_path, known_constraints, objectives, device=device, hardware=hardware)
        else:
            raise ValueError(f"Unknown algorithm: {algorithm}")

        self.max_observations = max_observations
        self.hardware = hardware
        self.model_name = f"{model}-{quant}"
        self.model = f"./../models/{model}-{quant}.gguf"
        self.parallel_degree = parallel_degree
        self.current_observations = 0
        self.objectives = objectives

        # History and metrics
        self.history = {'configs': [], 'performance': []}
        self.iter_hv: List[float] = []
        self.bounds = self._load_metric_bounds(f"bounds/{self.hardware}/{self.model_name}.json")
        self.hv_calc = HypervolumeCalculator(self.bounds)
        self.pareto_front: List[Tuple[Dict, Dict]] = []

    def run_workflow(self):
        """
        Run the optimization loop:
          - Initialize evaluations (special case for GA/Default)
          - Iterate until max_observations is reached
          - Track Pareto front and hypervolume per iteration
        """
        if isinstance(self.tuner, DefaultTuner):
            self._run_default_config()
            return
        elif isinstance(self.tuner, GeneticAlgorithmTuner):
            initial_samples = self.tuner.population
            initial_perfs = self._evaluate_configs(initial_samples, init_model=True)
            self.tuner.initialize_with_performance(initial_perfs)
        else:
            initial_samples = self.tuner.suggest_configurations(self.parallel_degree)
            initial_perfs = self._evaluate_configs(initial_samples, init_model=True)

        self.update_pareto_front()
        hv = self._compute_hypervolume()
        self.iter_hv.append(hv)

        while self.current_observations < self.max_observations:
            suggested_configs = self.tuner.suggest_configurations(self.parallel_degree)
            new_perfs = self._evaluate_configs(suggested_configs)
            self.tuner.update(suggested_configs, new_perfs)

            self.update_pareto_front()
            hv = self._compute_hypervolume()
            self.iter_hv.append(hv)

            print(
                f"Iteration {self.current_observations // self.parallel_degree + 1} "
                f"Observations: {self.current_observations}/{self.max_observations}"
            )

        # Optional: visualize HV curve
        # self._plot_hv_over_iterations()

    def _load_metric_bounds(self, path: str) -> Dict[str, Tuple[float, float]]:
        """
        Load metric bounds from JSON: {"metric": {"min": x, "max": y}, ...}
        Returns {"metric": (min, max)} with float values. Skips invalid entries.
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
                lo = float(lo)
                hi = float(hi)
            except (TypeError, ValueError):
                continue
            out[metric] = (lo, hi)
        return out

    def _run_default_config(self):
        """
        Evaluate a single default configuration, then update HV tracking.
        """
        if self.model_name in ("qwen3-4b-q4", "qwen3-4b-q8"):
            default_config = {"gpu-layers": 36}
        else:
            default_config = {"gpu-layers": 32}
        _ = self._evaluate_configs([default_config])
        self.update_pareto_front()
        hv = self._compute_hypervolume()
        self.iter_hv.append(hv)

    def _evaluate_configs(self, configs: List[Dict], init_model: bool = False) -> List[Dict]:
        """
        Evaluate a batch of configurations using the tuner's executor.
        Returns a list of perf dicts matching `self.objectives` keys.
        """
        perf_results: List[Dict] = []

        for config in configs:
            try:
                print(config)
                result = self.tuner.executor.run_server_performance_test(config, model_path=self.model)
                perf = {metric: result[metric] for metric in self.objectives}
            except ValueError:
                # Skip invalid runs
                continue

            print(f"perf: {perf}")

            # Log and count observation
            self.history['configs'].append(config)
            self.history['performance'].append(perf)
            self.current_observations += 1
            perf_results.append(perf)

            # Early stop if reaching observation budget
            if self.current_observations >= self.max_observations:
                break

        return perf_results

    def update_pareto_front(self):
        """
        Incrementally maintain the Pareto front based on recorded history.
        """
        new_front: List[Tuple[Dict, Dict]] = []

        for candidate in zip(self.history['configs'], self.history['performance']):
            dominated = False
            for front_sol in new_front:
                if self._dominates(front_sol[1], candidate[1]):
                    dominated = True
                    break
            if not dominated:
                new_front = [sol for sol in new_front if not self._dominates(candidate[1], sol[1])]
                new_front.append(candidate)

        self.pareto_front = new_front

    def _dominates(self, a: Dict[str, float], b: Dict[str, float]) -> bool:
        """
        Return True if point `a` Pareto-dominates `b` under `self.objectives`.
        objectives: {metric: 'min'|'max'}
        """
        better_in_any = False
        for obj, direction in self.objectives.items():
            if direction == 'min':
                if a[obj] > b[obj]:
                    return False
                if a[obj] < b[obj]:
                    better_in_any = True
            else:
                if a[obj] < b[obj]:
                    return False
                if a[obj] > b[obj]:
                    better_in_any = True
        return better_in_any

    def _compute_hypervolume(self) -> float:
        """
        Compute normalized hypervolume for current Pareto front.
        """
        if not self.history['performance']:
            return 0.0

        pareto_front = [t[1] for t in self.pareto_front]
        if not pareto_front:
            return 0.0

        hv = self.hv_calc.compute(pareto_front)
        print(f"Current Hypervolume: {hv:.4f}")
        return hv

    def _convert_to_serializable(self, obj: Any) -> Any:
        """
        Convert numpy scalar types and nested structures to JSON-serializable Python types.
        """
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

    def save_pareto_front_and_hv(self, method: str, model: str):
        """
        Save Pareto front and hypervolume progression to disk.
        """
        if self.objectives == 1:
            print("Only multi-objective Pareto fronts are supported.")
            return

        if not hasattr(self, 'pareto_front'):
            print("Pareto front not available.")
            return

        pareto_serializable = [
            {
                "config": self._convert_to_serializable(config),
                "perf": self._convert_to_serializable(perf)
            }
            for config, perf in self.pareto_front
        ]

        pf_path = f"pareto_fronts/{self.hardware}/{model}-{method}.json"
        with open(pf_path, 'w', encoding='utf-8') as f:
            json.dump(pareto_serializable, f, indent=2)
        print(f"Saved Pareto front to {pf_path}")

        hv_path = f"hv_progress/{self.hardware}/{model}-{method}.json"
        with open(hv_path, "w") as f:
            json.dump(self.iter_hv, f, indent=4)
        print(f"Saved HV progress to {hv_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Llama Configuration Optimizer')
    parser.add_argument('--device', type=str, choices=['cpu', 'gpu'], default='gpu',
                        help='Processing device (cpu or gpu)')
    parser.add_argument('--hardware', type=str, choices=['rtx3060', 'rtx4090', 'm4', 'orin'], default='rtx3060',
                        help='Processing hardware')
    parser.add_argument('--method', type=str, choices=['Default', 'GA', 'SCOOT', 'CBO'], default='SCOOT',
                        help='Optimization method')
    parser.add_argument('--model', type=str, choices=['qwen3-4b', 'phimoe-mini'], default='qwen3-4b',
                        help='Model family')
    parser.add_argument('--quant', type=str, choices=['q4', 'q8'], default='q4',
                        help='Quantization level')
    args = parser.parse_args()

    if args.device == 'gpu':
        objectives = {'tps_avg': 'max', 'gpu_avg': 'min'}
    else:
        objectives = {'tps_avg': 'max', 'mem_avg': 'min'}

    print("=======START========")
    print(f"model: {args.model}, quant: {args.quant}, hardware: {args.hardware}, method: {args.method}")

    workflow = BaselineWorkflow(
        parameters_path="knobs_files/knobs_raw.json",
        known_constraints=[],
        objectives=objectives,
        algorithm=args.method,
        max_observations=50,
        parallel_degree=5,
        device=args.device,
        hardware=args.hardware,
        model=args.model,
        quant=args.quant
    )
    workflow.run_workflow()
    workflow.save_pareto_front_and_hv(method=args.method, model=f"{args.model}-{args.quant}")
