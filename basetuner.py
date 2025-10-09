# basetuner.py
from abc import ABC, abstractmethod
from typing import List, Dict, Any
import numpy as np
from knobs import Knobs
from llama_executor import LlamaExecutor
import json
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern
from scipy.stats import norm


class BaseTuner(ABC):
    """Abstract base class for all tuners."""

    def __init__(self, parameters_path: str, known_constraints: List[str], objectives: List[str], device: str, hardware: str):
        self.parameters = Knobs(parameters_path, 5, random=False).knobs
        self.param_names = list(self.parameters.keys())
        self.objectives = objectives
        self.known_constraints = known_constraints
        self.param_types = {name: param["type"] for name, param in self.parameters.items()}
        self.device = device
        print(self.device)
        self.executor = LlamaExecutor(self.param_types, device=self.device, hardware=hardware)

    def _load_parameters(self, path: str) -> List[Dict]:
        """Load parameter definitions from a file."""
        with open(path, 'r') as f:
            return json.load(f)

    @abstractmethod
    def suggest_configurations(self, k: int) -> List[Dict]:
        """Generate k candidate configurations."""
        pass

    @abstractmethod
    def update(self, configs: List[Dict], performances: List[float]):
        """Update tuner state with new data."""
        pass


class DefaultTuner(BaseTuner):
    def suggest_configurations(self, k: int) -> List[Dict]:
        pass

    def update(self, configs: List[Dict], performances: List[float]):
        pass


class RandomTuner(BaseTuner):
    """Simple random search tuner."""

    def suggest_configurations(self, k: int) -> List[Dict]:
        return self.executor.generate_configs_fixed(self.parameters, n_samples=k)

    def update(self, configs: List[Dict], performances: List[float]):
        # No internal state to update for random sampling
        pass


class GeneticAlgorithmTuner(BaseTuner):
    """Genetic Algorithm tuner."""

    def __init__(self, parameters_path: str, known_constraints: List[str], objectives: List[str],
                 device: str, hardware: str, population_size: int = 5, mutation_rate: float = 0.1):
        super().__init__(parameters_path, known_constraints, objectives, device, hardware)
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.population = self._initialize_population()
        self.fitness = None
        self.device = device

    def _initialize_population(self) -> List[Dict]:
        return self.executor.generate_configs_fixed(self.parameters, self.population_size)

    def initialize_with_performance(self, performances: List[Dict]):
        """Initialize population fitness after first evaluation."""
        valid_perf = [perf['tps_avg'] for perf in performances if perf is not None]
        if len(valid_perf) != len(self.population):
            raise ValueError("Performance list length does not match population size.")
        self.fitness = valid_perf

    def suggest_configurations(self, k: int) -> List[Dict]:
        if self.fitness is None:
            raise RuntimeError("Fitness not initialized. Call initialize_with_performance() first.")
        parents = self._tournament_selection(k)
        offspring = self._crossover_and_mutate(parents)
        return offspring

    def _tournament_selection(self, k: int) -> List[Dict]:
        """Select parents using tournament selection."""
        selected = []
        for _ in range(k):
            candidates = np.random.choice(len(self.population), size=2, replace=False)
            best_idx = candidates[np.argmax([self.fitness[i] for i in candidates])]
            selected.append(self.population[best_idx])
        return selected

    def _crossover_and_mutate(self, parents: List[Dict]) -> List[Dict]:
        """Perform crossover and mutation to produce offspring."""
        offspring = []
        for i in range(0, len(parents), 1):
            p1 = parents[i]
            p2 = parents[i + 1] if i + 1 < len(parents) else p1
            child = {}
            for name, param in self.parameters.items():
                child[name] = p1[name] if np.random.rand() > 0.5 else p2[name]
                if np.random.rand() < self.mutation_rate:
                    child[name] = self._mutate_param(param)
            child = self.executor.handle_dependency(child)
            offspring.append(child)
        return offspring

    def _mutate_param(self, param_config: Dict) -> Any:
        """Randomly mutate a parameter."""
        t = param_config['type']
        if t == 'boolean':
            return not param_config.get('default', False)
        elif t == 'integer':
            return np.random.randint(param_config['values']['min'], param_config['values']['max'] + 1)
        elif t == 'enum':
            return np.random.choice(param_config['values'])
        elif t == 'float':
            return np.random.uniform(param_config['values']['min'], param_config['values']['max'])
        else:
            return param_config.get('default', None)

    def update(self, configs: List[Dict], performances: List[Dict]):
        """Update population and fitness after evaluation."""
        perf_values = [perf['tps_avg'] for perf in performances if perf is not None]
        if len(perf_values) != len(configs):
            raise ValueError("Performance count does not match configs count.")
        self.population.extend(configs)
        self.fitness.extend(perf_values)
        top_indices = np.argsort(self.fitness)[::-1][:self.population_size]
        self.population = [self.population[i] for i in top_indices]
        self.fitness = [self.fitness[i] for i in top_indices]


class ConstrainedBayesTuner(BaseTuner):
    """Bayesian Optimization with resource constraints."""

    def __init__(self, parameters_path, known_constraints, objectives, device, hardware, lambda_tps=97, lambda_pps=300):
        super().__init__(parameters_path, known_constraints, objectives, device, hardware)
        self.X = []
        self.y_res = []
        self.y_tps = []
        self.y_pps = []
        self.lambda_tps = lambda_tps
        self.lambda_pps = lambda_pps
        self.resource_metric = 'gpu_avg' if device == 'gpu' else 'mem_avg'
        self.gp_res = GaussianProcessRegressor(kernel=Matern(nu=2.5))
        self.gp_tps = GaussianProcessRegressor(kernel=Matern(nu=2.5))
        self.gp_pps = GaussianProcessRegressor(kernel=Matern(nu=2.5))

    def suggest_configurations(self, k: int) -> List[Dict]:
        """Suggest configurations using constrained expected improvement."""
        if len(self.X) < 5:
            return self.executor.generate_configs(self.parameters, n_samples=k)

        candidates = self.executor.generate_configs(self.parameters, n_samples=100)
        x = [self._encode_config(c) for c in candidates]
        cei_scores = self._compute_cei(x)
        top_indices = np.argsort(cei_scores)[-k:][::-1]
        return [candidates[i] for i in top_indices]

    def update(self, configs: List[Dict], performances: List[Dict]):
        """Update GP models with new data."""
        for config, perf in zip(configs, performances):
            if perf is None:
                continue
            self.X.append(self._encode_config(config))
            self.y_res.append(perf[self.resource_metric])
            self.y_tps.append(perf['tps_avg'])

        self.gp_res.fit(self.X, self.y_res)
        self.gp_tps.fit(self.X, self.y_tps)

    def _compute_cei(self, X: np.ndarray) -> np.ndarray:
        """Compute constrained expected improvement."""
        mu_res, sigma_res = self.gp_res.predict(X, return_std=True)
        mu_tps, sigma_tps = self.gp_tps.predict(X, return_std=True)

        feasible_mask = (np.array(self.y_tps) >= self.lambda_tps)
        f_best = np.min(np.array(self.y_res)[feasible_mask]) if any(feasible_mask) else np.min(self.y_res)

        imp = f_best - mu_res
        Z = imp / sigma_res
        ei = imp * norm.cdf(Z) + sigma_res * norm.pdf(Z)
        ei[sigma_res == 0.0] = 0.0

        p_tps = 1.0 - norm.cdf((self.lambda_tps - mu_tps) / sigma_tps)
        cei = ei * p_tps
        return cei

    def _encode_config(self, config):
        """Encode configuration into numeric vector for GP input."""
        encoded = []
        for name in self.param_names:
            val = config[name]
            param_info = self.parameters[name]
            if self.param_types[name] == 'integer':
                encoded.append((val - param_info['values']['min']) / (param_info['values']['max'] - param_info['values']['min']))
            elif self.param_types[name] == 'float':
                encoded.append((val - param_info['values']['min']) / (param_info['values']['max'] - param_info['values']['min']))
            elif self.param_types[name] == 'enum':
                options = param_info['values']
                encoded.append(options.index(val) / (len(options) - 1))
            elif self.param_types[name] == 'boolean':
                encoded.append(1.0 if val else 0.0)
        return np.array(encoded)
