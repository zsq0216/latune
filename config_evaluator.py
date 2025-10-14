import time
from typing import List, Dict, Tuple
import numpy as np
from llama_executor import LlamaExecutor
from knobs import Knobs
import json
from pathlib import Path
import signal
from threading import Thread, Event
from system_load_simulator import SystemLoadGenerator
import sys
import subprocess


class ConfigEvaluator:
    def __init__(self,
                 parameters_path: str,
                 pareto_front_path: str,
                 device: str = "gpu",
                 model_path: str = '',
                 parameter_num=5):
        """
        Evaluate and monitor optimal configurations under system resource constraints.
        """
        self.parameters = Knobs(parameters_path, parameter_num, random=False).knobs
        self.param_types = {name: params['type'] for name, params in self.parameters.items()}
        self.device = device
        self.executor = LlamaExecutor(self.param_types, model_path=model_path, device=device)
        self.model_path = model_path
        self.fluctuation_detector = SystemLoadGenerator()
        self.fluctuation_event = Event()
        self.performance_log = []
        self.start_time = time.time()
        self.stream_thread = None
        self.pareto_front = self._load_pareto_front(pareto_front_path)

    def _load_pareto_front(self, filepath: str):
        """Load Pareto front data from file."""
        filepath = Path(filepath)
        if not filepath.exists():
            print(f"File {filepath} not found.")
            return

        with open(filepath, 'r', encoding='utf-8') as f:
            pareto_data = json.load(f)

        pareto_front = [(item["config"], item["perf"]) for item in pareto_data]
        return pareto_front

    def _handle_exit(self, sig, frame):
        print("\nInterrupt signal received. Cleaning up...")
        self._terminate_all()
        with open('performance_log.json', 'w') as f:
            json.dump(self.performance_log, f, indent=4)
        sys.exit(0)

    def _get_pareto_best(self, budget, ablation=False) -> Dict:
        """Get the best Pareto configuration under resource budget."""
        if self.device == 'gpu':
            valid_pareto = [
                (cfg, perf) for cfg, perf in self.pareto_front
                if perf['gpu_avg'] <= budget['gpu_avail']
            ]
        else:
            valid_pareto = [
                (cfg, perf) for cfg, perf in self.pareto_front
                if perf['mem_avg'] <= budget['mem_avail']
            ]

        if not valid_pareto:
            raise ValueError("No valid Pareto configurations under the given budget.")

        valid_perfs = [perf for _, perf in valid_pareto]
        best_idx, best_score = self._evaluate_pareto(valid_perfs, ablation=ablation)
        return valid_pareto[best_idx][0]

    def _get_pareto_best_no_budget(self) -> List[Dict]:
        """Get the best Pareto configuration without resource constraints."""
        perfs = [sample[1] for sample in self.pareto_front]
        best_idx, best_score = self._evaluate_pareto(perfs)
        return self.pareto_front[best_idx][0]

    def _evaluate_pareto(self, perfs, ablation=False) -> Tuple[int, float]:
        """Evaluate Pareto front and return index and score of the best configuration."""
        tps = np.array([perf['tps_avg'] for perf in perfs])
        if self.device == 'gpu':
            gpu = np.array([perf['gpu_avg'] for perf in perfs])

        def normalize(arr, is_benefit=True):
            """Normalize metrics to [0, 1]."""
            if is_benefit:
                return (arr - np.min(arr)) / (np.max(arr) - np.min(arr))
            else:
                return (np.max(arr) - arr) / (np.max(arr) - np.min(arr))

        tps_norm = normalize(tps, is_benefit=True)
        if self.device == 'gpu':
            gpu_norm = normalize(gpu, is_benefit=False)

        w_tps = 1
        w_gpu = 0

        scores = w_tps * tps_norm + w_gpu * gpu_norm
        if ablation:
            scores = tps_norm

        best_idx = np.argmax(scores)
        best_score = scores[best_idx]
        return best_idx, best_score

    def run_monitoring_cycle(self, duration=50, loop=False):
        """Monitoring cycle: select best config → start test → wait → stop."""
        self.fluctuation_event.clear()
        self.executor.stop_event = Event()

        budget = self.fluctuation_detector.get_available_resources()
        print(f"Resource budget: {budget}")
        best_config = self._get_pareto_best(budget)
        print(f"Starting service with best configuration: {best_config}")

        self.stream_thread = Thread(
            target=self._start_streaming,
            args=(best_config, self.model_path),
            daemon=True
        )
        self.stream_thread.start()

        if loop:
            while not self.fluctuation_event.is_set():
                time.sleep(2)
        else:
            time.sleep(duration)

        self._terminate_all()

    def adjust_monitoring(self, duration=50):
        print("\nManual monitoring adjustment...")
        self.run_monitoring_cycle(duration=duration)

    def adaptive_monitoring_loop(self):
        print("\nAdaptive monitoring loop started...")
        signal.signal(signal.SIGINT, self._handle_exit)

        while True:
            self._start_fluctuation_monitor()
            self.run_monitoring_cycle(duration=50, loop=True)

    def fixed_monitoring(self, duration=50):
        """Run monitoring using best fixed configuration (no adaptation)."""
        best_config = self._get_pareto_best_no_budget()
        self.stream_thread = Thread(
            target=self._start_streaming,
            args=(best_config, self.model_path),
            daemon=True
        )
        self.stream_thread.start()
        time.sleep(duration)
        self._terminate_all()

    def evaluate_instance(self, method='latune', model='qwen3-4b-q4'):
        """Evaluate single instance based on selected tuning method."""
        model = f"./../models/{model}.gguf"
        if method not in ["Default", "GA", "CBO", "scoot", "latune"]:
            raise ValueError("Invalid method. Choose from [Default, GA, CBO, scoot, latune]")

        budget = self.fluctuation_detector.get_available_resources()
        print(f"Resource budget: {budget}")
        best_config = self._get_pareto_best(budget)
        print(f"Starting service with best configuration: {best_config}")
        return self.executor.run_server_performance_test(config=best_config, model_path=model)

    def _start_streaming(self, config, model_path):
        """Start continuous streaming and log performance reports."""
        for report in self.executor.run_server_performance_test_streaming(config, model_path):
            print("Collected metrics:", report)
            timestamp = time.time() - self.start_time
            self.performance_log.append({"perf": report, "timestamp": timestamp})
            if self.fluctuation_event.is_set() or self.executor.stop_event.is_set():
                break

    def _start_fluctuation_monitor(self):
        """Start background resource fluctuation monitoring."""
        def notify(changed, message):
            print(message)
            if changed:
                self.fluctuation_event.set()

        monitor_thread = Thread(
            target=self.fluctuation_detector.detect_fluctuation_continuous,
            kwargs={
                'resource_type': 'gpu',
                'duration': 10,
                'interval': 10,
                'threshold': 0.1,
                'notify_func': notify
            },
            daemon=True
        )
        monitor_thread.start()

    def _terminate_all(self):
        """Stop all running services and monitoring threads."""
        print("Terminating current services and threads...")
        self.executor.stop_event.set()
        self.stream_thread.join(timeout=3)
        self.fluctuation_detector.stop_event.set()
        self.fluctuation_event.clear()

        if hasattr(self.executor, 'stop_event'):
            print("Stopping performance collection thread...")
            self.executor.stop_event.set()

        if hasattr(self.executor, 'server_process'):
            print("Terminating server process...")
            self.executor.server_process.terminate()
            try:
                self.executor.server_process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.executor.server_process.kill()


if __name__ == "__main__":
    config_evaluator = ConfigEvaluator(
        parameters_path="ranked_knobs_files/rank_server_knobs_gpu.json",
        pareto_front_path="pareto_fronts/pareto_front_GA.json",
        device="gpu",
        model_path="./../../models/qwen2.5-1.5b-instruct-fp16.gguf"
    )
    # config_evaluator.adaptive_monitoring_loop()
