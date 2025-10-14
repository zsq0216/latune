import numpy as np
from sklearn.ensemble import RandomForestClassifier
from scipy.stats import norm
from sobol_seq import i4_sobol_generate  # Requires: pip install sobol_seq
from knobs import Knobs
from surrogate_model import SurrogateModel
from pymoo.indicators.hv import HV  # Requires: pip install pymoo
from typing import List, Dict, Tuple
import random
import json


class LaTune:
    """
    LaTune: A multi-objective auto-tuning framework that combines surrogate modeling,
    Bayesian acquisition, and Pareto optimization.

    Attributes:
        parameters (dict): Parameter definitions loaded from JSON.
        delta (float): Step size or exploration factor.
        rf (RandomForestClassifier): Classifier for feasibility prediction.
        surrogate (SurrogateModel): Surrogate model for objective prediction.
        observations (list): History of observed configurations and objective values.
        pareto_front (list): List of current Pareto-optimal solutions.
    """

    def __init__(self, parameters_path, objectives, delta_init=0.5, num_gpus=4):
        """
        Initialize LaTune with parameter definitions and optimization objectives.

        Args:
            parameters_path (str): Path to the JSON file defining tunable parameters.
            objectives (dict): Mapping of objective names to direction ('min' or 'max').
            delta_init (float, optional): Initial exploration delta. Defaults to 0.5.
            num_gpus (int, optional): Number of GPUs available. Defaults to 4.
        """
        self.parameters = Knobs(parameters_path, 5, random=False).knobs  # Load top-N parameters
        self.delta = delta_init
        self.consecutive_feasible = 0
        self.rf = RandomForestClassifier(n_estimators=100)
        self.X = []                     # Encoded configuration feature vectors
        self.y = []                     # Feasibility labels (0/1)
        self.observations = []          # List of (config, objective_values)
        self.iteration = 1
        self.num_gpus = num_gpus

        # Parameter encoding metadata
        self.param_names = list(self.parameters.keys())
        self.param_types = {n: p['type'] for n, p in self.parameters.items()}

        # Surrogate model setup
        self.objectives = objectives
        self.num_objectives = len(objectives)
        self.surrogate = SurrogateModel(num_objectives=self.num_objectives)
        self.observed_perf = {obj: [] for obj in objectives}
        self.pareto_front = []

    # -------------------------------------------------------------------------
    # Data encoding and surrogate model
    # -------------------------------------------------------------------------

    def _encode_config(self, config):
        """
        Encode a configuration dictionary into a normalized numeric vector.

        Args:
            config (dict): Configuration mapping parameter names to values.

        Returns:
            np.ndarray: Normalized feature vector.
        """
        encoded = []
        for name in self.param_names:
            val = config[name]
            param_info = self.parameters[name]
            ptype = self.param_types[name]

            if ptype in ('integer', 'float'):
                min_val = param_info['values']['min']
                max_val = param_info['values']['max']
                encoded.append((val - min_val) / (max_val - min_val))
            elif ptype == 'enum':
                options = param_info['values']
                encoded.append(options.index(val) / (len(options) - 1))
            elif ptype == 'boolean':
                encoded.append(1.0 if val else 0.0)

        return np.array(encoded)

    def update_surrogate(self, window_size: int = 5):
        """
        Train or update the surrogate model using the most recent observations.

        Args:
            window_size (int): Number of latest samples used for retraining.
        """
        recent_size = min(len(self.observations), window_size)
        recent_data = self.observations[-recent_size:]

        X = [self._encode_config(c) for c, _ in recent_data]
        y_list = [[perf[obj] for _, perf in recent_data] for obj in self.objectives]

        self.surrogate.fit(X, y_list)

    def save_surrogate(self, filename):
        """Save the trained surrogate model to disk."""
        self.surrogate.save_model(filename)

    # -------------------------------------------------------------------------
    # Sampling and configuration generation
    # -------------------------------------------------------------------------

    def generate_initial_samples(self, n_samples):
        """
        Generate initial configurations using Sobol sequence sampling.

        Args:
            n_samples (int): Number of configurations to generate.

        Returns:
            list[dict]: List of sampled configurations.
        """
        dim = len(self.parameters)
        sobol_points = i4_sobol_generate(dim, n_samples)
        configs = []

        for point in sobol_points:
            config = {}
            for i, name in enumerate(self.param_names):
                param_info = self.parameters[name]
                ptype = self.param_types[name]

                if ptype == 'integer':
                    min_val = param_info['values']['min']
                    max_val = param_info['values']['max']
                    config[name] = int(min_val + point[i] * (max_val - min_val))
                elif ptype == 'float':
                    min_val = param_info['values']['min']
                    max_val = param_info['values']['max']
                    config[name] = min_val + point[i] * (max_val - min_val)
                elif ptype == 'enum':
                    options = param_info['values']
                    idx = int(point[i] * len(options))
                    config[name] = options[min(idx, len(options) - 1)]
                elif ptype == 'boolean':
                    config[name] = point[i] > 0.5

            config = self.handle_dependency(config)
            configs.append(config)

        return configs

    def generate_configs(self, n_samples=100):
        """
        Generate random configurations uniformly across parameter ranges.

        Args:
            n_samples (int, optional): Number of samples to generate. Defaults to 100.

        Returns:
            list[dict]: Randomly generated configuration list.
        """
        configs = []
        for _ in range(n_samples):
            config = {}
            for name, info in self.parameters.items():
                ptype = info['type']
                if ptype == 'boolean':
                    config[name] = np.random.choice([True, False])
                elif ptype == 'integer':
                    config[name] = np.random.randint(info['values']['min'], info['values']['max'] + 1)
                elif ptype == 'enum':
                    config[name] = np.random.choice(info['values'])
                elif ptype == 'float':
                    config[name] = np.random.uniform(info['values']['min'], info['values']['max'])
            configs.append(config)
        return configs

    def load_configs_from_history(self, json_path):
        """
        Load configurations from a history JSON file and fill in missing parameters.

        Args:
            json_path (str): Path to the configuration history file.

        Returns:
            list[dict]: Completed configurations.
        """
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            data = [data]

        all_configs = []
        for item in data:
            raw_config = item.get("config", {})
            cfg = {}
            for name, info in self.parameters.items():
                if name in raw_config:
                    cfg[name] = raw_config[name]
                else:
                    # Fill missing parameters with random valid values
                    ptype = self.param_types[name]
                    vals = info["values"]
                    if ptype == "integer":
                        cfg[name] = random.randint(vals["min"], vals["max"])
                    elif ptype == "float":
                        cfg[name] = random.uniform(vals["min"], vals["max"])
                    elif ptype == "enum":
                        cfg[name] = random.choice(vals)
                    elif ptype == "boolean":
                        cfg[name] = random.choice([True, False])
            all_configs.append(cfg)

        return all_configs

    def handle_dependency(self, config):
        """
        Handle inter-parameter dependencies (example: grp-attn-w = k * grp-attn-n).

        Args:
            config (dict): Configuration dictionary.

        Returns:
            dict: Updated configuration with valid dependencies applied.
        """
        if 'grp-attn-n' not in config:
            return config
        max_multiplier = 2048 // config['grp-attn-n']  # 2048 = upper bound for grp-attn-w
        config['grp-attn-w'] = config['grp-attn-n'] * np.random.randint(1, max_multiplier + 1)
        return config

    # -------------------------------------------------------------------------
    # Acquisition and optimization
    # -------------------------------------------------------------------------

    def acquisition_ucb(self, config):
        """
        Upper Confidence Bound (UCB) acquisition function.

        Args:
            config (dict): Configuration to evaluate.

        Returns:
            float: UCB acquisition score.
        """
        x = self._encode_config(config)
        mu, var = self.surrogate.predict([x])
        sigma = np.sqrt(var[0])
        beta = 2 * np.log((self.iteration**2 * np.pi**2) / (6 * 0.1))
        return mu[0] + beta * sigma

    def suggest_configurations(self, k=1):
        """
        Suggest top-K configurations using acquisition function ranking.

        Args:
            k (int, optional): Number of configurations to return. Defaults to 1.

        Returns:
            list[dict]: Suggested configurations.
        """
        candidates = self.generate_configs(100)
        scores = []

        for config in candidates:
            if self.num_objectives == 1:
                score = self.acquisition_ucb(config)
            else:
                score = self.acquisition_ehvi(config)
            scores.append(score)

        top_indices = np.argsort(scores)[-k:]
        return [candidates[i] for i in top_indices]

    # -------------------------------------------------------------------------
    # Pareto optimization
    # -------------------------------------------------------------------------

    def update_pareto_front(self):
        """Update the Pareto front based on observed configurations."""
        new_front = []
        feasible_solutions = [(cfg, perf) for cfg, perf in self.observations]

        for candidate in feasible_solutions:
            dominated = any(self._dominates(f[1], candidate[1]) for f in new_front)
            if not dominated:
                new_front = [sol for sol in new_front if not self._dominates(candidate[1], sol[1])]
                new_front.append(candidate)

        self.pareto_front = new_front

    def _dominates(self, a: Dict[str, float], b: Dict[str, float]) -> bool:
        """Check if solution a dominates solution b."""
        better = False
        for obj, direction in self.objectives.items():
            if direction == 'min':
                if a[obj] > b[obj]:
                    return False
                if a[obj] < b[obj]:
                    better = True
            else:
                if a[obj] < b[obj]:
                    return False
                if a[obj] > b[obj]:
                    better = True
        return better

    def set_reference_point(self):
        """Set the reference point for hypervolume calculation."""
        ref_point = []
        for obj, direction in self.objectives.items():
            values = [perf[obj] for _, perf in self.observations]
            ref_val = max(values) if direction == "min" else min(values)
            ref_point.append(ref_val)
        self.reference_point = ref_point

    def get_pareto_front(self) -> List[Tuple[Dict, Dict]]:
        """Return the current Pareto front (config + objectives)."""
        return self.pareto_front

    def acquisition_ehvi(self, config):
        """
        Expected Hypervolume Improvement (EHVI) acquisition function (Monte Carlo approximation).
        """
        x = self._encode_config(config)
        preds = self.surrogate.predict([x])
        means = np.array([mu[0] for mu, _ in preds])
        stds = np.array([sigma[0] for _, sigma in preds])

        if not hasattr(self, 'reference_point') or not self.pareto_front:
            return 0.0

        front_y = np.array([[perf[obj] for obj in self.objectives] for _, perf in self.pareto_front])
        obj_dirs = [1 if self.objectives[obj] == 'min' else -1 for obj in self.objectives]
        front_y *= obj_dirs
        ref_point = np.array(self.reference_point) * obj_dirs
        mu_scaled = means * obj_dirs
        std_scaled = stds

        hv = HV(ref_point=ref_point)
        samples = np.random.normal(loc=mu_scaled, scale=std_scaled, size=(128, self.num_objectives))
        hvs = [hv.do(np.vstack([front_y, s])) for s in samples]
        current_hv = hv.do(front_y)
        return np.mean([h - current_hv for h in hvs])

    # -------------------------------------------------------------------------
    # Evaluation
    # -------------------------------------------------------------------------

    def evaluate_pareto(self, perfs):
        """
        Evaluate Pareto candidates using weighted normalized objectives.

        Args:
            perfs (list[dict]): Performance metrics (tps, gpu, etc.).

        Returns:
            tuple: (best_index, best_score)
        """
        tps = np.array([p['tps_avg'] for p in perfs])
        gpu = np.array([p['gpu_p95'] for p in perfs])

        def normalize(arr, benefit=True):
            return (arr - np.min(arr)) / (np.max(arr) - np.min(arr)) if benefit \
                else (np.max(arr) - arr) / (np.max(arr) - np.min(arr))

        tps_norm = normalize(tps, benefit=True)
        gpu_norm = normalize(gpu, benefit=False)

        w_tps, w_gpu = 0.7, 0.3
        scores = w_tps * tps_norm + w_gpu * gpu_norm

        best_idx = np.argmax(scores)
        return best_idx, scores[best_idx]


# -------------------------------------------------------------------------
# Example Usage
# -------------------------------------------------------------------------

if __name__ == "__main__":
    parameters_path = "path/to/parameters.json"
    objectives = {'tps_avg': 'max', 'gpu_p95': 'min'}

    tuner = LaTune(parameters_path, objectives)

    # Generate initial Sobol samples
    initial_samples = tuner.generate_initial_samples(10)
    print("Initial Samples:", initial_samples)

    # Train/update surrogate
    tuner.update_surrogate()

    # Suggest configurations
    suggested_configs = tuner.suggest_configurations(k=5)
    print("Suggested Configurations:", suggested_configs)
