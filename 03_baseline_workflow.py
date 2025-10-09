import time
from typing import List, Dict, Tuple, Any
import numpy as np
from latune import LaTune
from llama_executor import LlamaExecutor
import argparse
import random
import json
from pathlib import Path
from hv_calculator import HypervolumeCalculator
import matplotlib.pyplot as plt


class TUNINGWorkflow:
    def __init__(
        self,
        parameters_path: str,
        objectives: Dict,
        max_observations: int = 30,
        parallel_degree: int = 5,
        device: str = "gpu",
        hardware: str = "rtx3060",
        model: str = "qwen3-4b",
        quant: str = "q4",
    ):
        """
        Orchestrates an end-to-end tuning loop using LaTune and a performance executor.

        Parameters
        ----------
        parameters_path : str
            Path to the JSON knobs/parameters definition.
        objectives : Dict
            Mapping of metric -> direction ('max' or 'min').
        max_observations : int
            Total evaluation budget.
        parallel_degree : int
            Number of configurations to suggest per iteration.
        device : str
            'cpu' or 'gpu' (forwarded to the executor).
        hardware : str
            Hardware label for bookkeeping/output paths.
        model : str
            Base model name (e.g., 'qwen3-4b').
        quant : str
            Quantization tag (e.g., 'q4', 'q8').
        """
        # Core components
        self.tuner = LaTune(parameters_path, objectives)
        self.objectives = objectives
        self.max_observations = max_observations
        self.parallel_degree = parallel_degree
        self.executor = LlamaExecutor(self.tuner.param_types, device=device, hardware=hardware)
        self.hardware = hardware
        self.model_name = f"{model}-{quant}"
        self.model = f"./../models/{model}-{quant}.gguf"

        # Workflow state tracking
        self.current_observations = 0
        self.history = {
            "configs": [],
            "performance": [],
        }
        self.iter_hv = []
        self.bounds = self._load_metric_bounds(f"bounds/{self.hardware}/{self.model_name}.json")
        self.hv_calc = HypervolumeCalculator(self.bounds)

    def run_workflow(self):
        """Run the full workflow: initialize, iterate suggestions/evaluations, update models, and stop on budget."""
        # Steps 1–2: initial design and evaluation
        initial_samples = self._generate_initial_samples()
        self._evaluate_configs(initial_samples, init_model=True)

        # Iterative optimization (Steps 3–9)
        while self.current_observations < self.max_observations:
            start_time = time.time()

            # Step 3: suggest new configurations
            suggested_configs = self.tuner.suggest_configurations(k=self.parallel_degree)

            # Step 4: evaluate new configurations
            new_perfs = self._evaluate_configs(suggested_configs)

            # Step 7: update the surrogate with the number of new observations
            self.tuner.update_surrogate(len(new_perfs))

            # Maintain Pareto front and reference point (used for HV)
            self.tuner.update_pareto_front()
            self.tuner.set_reference_point()

            # Iteration bookkeeping
            iteration_time = time.time() - start_time
            print(
                f"Iteration {self.tuner.iteration} completed. "
                f"Observations: {self.current_observations}/{self.max_observations} "
                f"Time: {iteration_time:.1f}s"
            )
            self.tuner.iteration = self.tuner.iteration + 1

            hv = self._compute_hypervolume()
            self.iter_hv.append(hv)

            if self.current_observations >= self.max_observations:
                self.tuner.save_surrogate(f"surrogate_models/{self.hardware}/{self.model_name}.pth")
                # self._plot_hv_over_iterations()
                break

    def _generate_initial_samples(self) -> List[Dict]:
        """Generate the initial design (Step 1)."""
        return self.tuner.generate_initial_samples(5)

    def _load_metric_bounds(self, path: str) -> Dict[str, Tuple[float, float]]:
        """
        Load per-metric bounds from JSON of the form:
            {"metric": {"min": x, "max": y}, ...}
        Returns:
            {"metric": (min, max)} with values cast to float. Invalid/missing entries are skipped.
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

    def _evaluate_configs(self, configs: List[Dict], init_model=False):
        """
        Evaluate a list of configurations via the executor and update the tuner/history.

        Parameters
        ----------
        configs : List[Dict]
            Configurations to evaluate.
        init_model : bool
            If True, also initializes/refreshes the surrogate and Pareto front after evaluation.

        Returns
        -------
        List[Dict]
            List of metric dicts (skips failed runs).
        """
        perf_results = []

        for config in configs:
            try:
                result = self.executor.run_server_performance_test(config, model_path=self.model)
                # Keep only the requested objectives
                perf = {metric: result[metric] for metric in self.objectives}
            except ValueError:
                # Skip invalid runs
                continue

            self.tuner.observations.append((config, perf))
            print(f"config: {config}, perf: {perf}")

            # Bookkeeping
            self.history["configs"].append(config)
            self.history["performance"].append(perf)
            self.current_observations += 1

            perf_results.append(perf)

            # Early stopping if budget is reached mid-batch
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

    def get_best_config(self) -> Dict:
        """
        Return the current best configuration.
        - Single objective: argmin/argmax on the target metric.
        - Multi-objective: pick from the Pareto front via the tuner's evaluation utility.
        """
        if len(self.tuner.objectives) == 1:
            return self._get_single_best()
        else:
            return self._get_pareto_best()

    def _get_single_best(self) -> Dict:
        """Select the best configuration for a single-objective run."""
        obj = list(self.objectives.keys())[0]
        direct = list(self.objectives.values())[0]
        perfs = [perf[obj] for perf in self.history["performance"]]
        best_idx = np.argmin(perfs) if direct == "min" else np.argmax(perfs)
        print(self.history["performance"][best_idx])
        return self.history["configs"][best_idx]

    def _get_pareto_best(self) -> List[Dict]:
        """Select a representative best configuration from the current Pareto front."""
        pareto_front = self.tuner.get_pareto_front()
        print(pareto_front)
        perfs = [sample[1] for sample in pareto_front]
        best_idx, best_score = self.tuner.evaluate_pareto(perfs)
        print(best_score)
        print(pareto_front[best_idx])
        return pareto_front[best_idx][0]

    def _convert_to_serializable(self, obj: Any) -> Any:
        """Convert numpy types to native Python types for JSON serialization."""
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
        """
        Compute the normalized hypervolume (HV) of the current Pareto front
        given the preloaded bounds (e.g., tps_avg ↑, gpu_avg ↓).
        """
        if not self.history["performance"]:
            return 0.0

        pareto_front = [t[1] for t in self.tuner.get_pareto_front()]
        if not pareto_front:
            return 0.0

        hv = self.hv_calc.compute(pareto_front)
        print(f"Current Hypervolume: {hv:.4f}")
        return hv

    def _plot_hv_over_iterations(self):
        """Plot the evolution of the best hypervolume across iterations."""
        plt.figure(figsize=(8, 5))
        plt.plot(range(1, len(self.iter_hv) + 1), self.iter_hv, marker="o")
        plt.xlabel("Iteration")
        plt.ylabel("Hypervolume")
        plt.title("Best Hypervolume Over Iterations")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig("hypervolume_progress_latune.png")
        plt.show()

    def save_pareto_front_and_hv(self, model: str):
        """
        Save the current Pareto front and the hypervolume progression to disk.

        Output files
        ------------
        - pareto_fronts/{hardware}/{model}-scoot.json
        - hv_progress/{hardware}/{model}-scoot.json
        """
        if self.objectives == 1:
            print("Saving the Pareto front is only supported for multi-objective runs.")
            return

        if not hasattr(self.tuner, "pareto_front"):
            print("No Pareto front is available to save.")
            return

        pareto_serializable = [
            {"config": self._convert_to_serializable(config), "perf": self._convert_to_serializable(perf)}
            for config, perf in self.tuner.pareto_front
        ]

        with open(f"pareto_fronts/{self.hardware}/{model}-scoot.json", "w", encoding="utf-8") as f:
            json.dump(pareto_serializable, f, indent=2)
        print(f"Pareto front saved to pareto_fronts/{self.hardware}/{model}-scoot.json")

        with open(f"hv_progress/{self.hardware}/{model}-scoot.json", "w") as f:
            json.dump(self.iter_hv, f, indent=4)
        print(f"Hypervolume progress saved to hv_progress/{self.hardware}/{model}-scoot.json")

    def load_pareto_front(self, filepath: str):
        """Load a Pareto front from disk and populate `self.tuner.pareto_front`."""
        filepath = Path(filepath)
        if not filepath.exists():
            print(f"File {filepath} does not exist.")
            return

        with open(filepath, "r", encoding="utf-8") as f:
            pareto_data = json.load(f)

        self.tuner.pareto_front = [(item["config"], item["perf"]) for item in pareto_data]
        print(f"Pareto front loaded from {filepath}.")

    def visualize_pareto_front(self):
        """
        Visualize the Pareto front for the 2-objective case (x = gpu_avg, y = tps_avg).
        Shows the plot and saves a PDF snapshot.
        """
        import matplotlib.pyplot as plt

        front = self.tuner.get_pareto_front()
        print(front)
        perfs = [sample[1] for sample in front]
        if len(self.tuner.objectives) != 2:
            print("Visualization supports the 2-objective case only.")
            return

        y = np.array([perf["tps_avg"] for perf in perfs])
        x = np.array([perf["gpu_avg"] for perf in perfs])

        plt.scatter(x, y, c="red", label="Pareto Front")
        plt.xlabel("gpu_avg")
        plt.ylabel("tps_avg")
        plt.title("Pareto Front Evolution")
        plt.legend()
        plt.show()
        # Save as PDF
        plt.savefig("tps_gpu_r3.pdf")


# Example usage
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Llama Configuration Optimizer")
    parser.add_argument("--device", type=str, choices=["cpu", "gpu"], default="gpu", help="Processing device (cpu or gpu)")
    parser.add_argument(
        "--hardware", type=str, choices=["rtx3060", "rtx4090", "m4", "orin"], default="rtx3060", help="Processing hardware"
    )
    parser.add_argument(
        "--model", type=str, choices=["qwen3-4b", "phimoe-mini"], default="phimoe-mini", help="Model family (e.g., qwen3-8b, phimoe-mini)"
    )
    parser.add_argument("--quant", type=str, choices=["q4", "q8"], default="q8", help="Quantization tag (q4, q8)")
    args = parser.parse_args()
    parameters_path = "knobs_files/knobs_raw.json"

    if args.device == "gpu":
        objectives = {"tps_avg": "max", "gpu_avg": "min"}
    else:
        objectives = {"tps_avg": "max", "mem_avg": "min"}

    print("======= START =======")
    print(f"Running SCOOT on {args.hardware} for model {args.model}-{args.quant}")

    # Initialize workflow
    workflow = TUNINGWorkflow(
        parameters_path=parameters_path,
        objectives=objectives,
        max_observations=50,
        parallel_degree=5,
        device=args.device,
        hardware=args.hardware,
        model=args.model,
        quant=args.quant,
    )

    # Run the complete workflow
    workflow.run_workflow()

    # Output results
    print("\n=== Tuning Results ===")
    print(f"Total evaluations: {len(workflow.history['configs'])}")
    print("Best configuration:")
    # print(workflow.get_best_config())

    workflow.save_pareto_front_and_hv(f"{args.model}-{args.quant}")
