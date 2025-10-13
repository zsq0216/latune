# basetuner.py
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Tuple, Optional
import numpy as np
from knobs import Knobs
from llama_executor import LlamaExecutor
import json
import os
import math
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern
from scipy.stats import norm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel as C, WhiteKernel as W
from surrogate_model import SurrogateModel
from sklearn.ensemble import RandomForestRegressor


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
    """
    Constrained Bayesian Optimization with meta-learning (Res + TPS only):
      - Objective: minimize resource usage (res)
      - Constraint: tps >= lambda_tps
      - Target task: sklearn Gaussian Process (Matern kernel)
      - Source tasks: GPyTorch SurrogateModels (loaded from .pth)
      - Knowledge transfer: mean = weighted average of target GP and source models (by similarity);
                           variance = from target GP only
    """

    def __init__(self, parameters_path, known_constraints, objectives, device, hardware, model_name,
                 lambda_tps=10):
        super().__init__(parameters_path, known_constraints, objectives, device, hardware)

        # Raw data buffers
        self.X: List[np.ndarray] = []
        self.y_res: List[float] = []
        self.y_tps: List[float] = []

        self.hardware = hardware
        self.model_name = model_name

        # Constraint threshold (original scale)
        self.lambda_tps = float(lambda_tps)

        # Resource metric name
        self.resource_metric = 'gpu_avg' if device == 'gpu' else 'mem_avg'

        # Target task GPs (trained on standardized targets)
        kernel = C(1.0, (1e-3, 1e3)) * Matern(nu=2.5) + W(1e-3, (1e-6, 1e-1))
        self.gp_res = GaussianProcessRegressor(kernel=kernel, normalize_y=False)
        self.gp_tps = GaussianProcessRegressor(kernel=kernel, normalize_y=False)

        # Target task normalization stats
        self._mu_res = 0.0; self._std_res = 1.0
        self._mu_tps = 0.0; self._std_tps = 1.0
        self._lam_tps_std = None

        # Historical source models and similarity weights
        self.source_models: List[SurrogateModel] = []
        self.source_weights: np.ndarray = None
        self._meta_enabled = False

        self._register_source_models()

    # ---------------- Public API ----------------

    def suggest_configurations(self, k: int) -> List[Dict]:
        """Score candidates via Constrained Expected Improvement (CEI) and return top k."""
        if len(self.X) < 5:
            return self.executor.generate_configs_fixed(self.parameters, n_samples=k)

        candidates = self.executor.generate_configs(self.parameters, n_samples=200)
        Xc = np.vstack([self._encode_config(c) for c in candidates])
        cei_scores = self._compute_cei(Xc)
        top_idx = np.argsort(cei_scores)[-k:][::-1]
        return [candidates[i] for i in top_idx]

    def update(self, configs: List[Dict], performances: List[Dict]):
        """
        Update GPs using (res, tps) only.
        - Normalize both targets.
        - Map tps threshold into standardized space.
        - Combine with pre-trained source models via similarity-weighted mean transfer.
        """
        # Collect new observations
        for cfg, perf in zip(configs, performances):
            if perf is None:
                continue
            self.X.append(self._encode_config(cfg))
            self.y_res.append(float(perf[self.resource_metric]))
            self.y_tps.append(float(perf['tps_avg']))

        if len(self.X) < 2:
            return

        X_np = np.vstack(self.X)
        y_res = np.array(self.y_res)
        y_tps = np.array(self.y_tps)

        # Normalize targets (z-score)
        def _fit_stats(v):
            mu = float(np.mean(v))
            std = float(np.std(v)) if np.std(v) > 1e-8 else 1.0
            return mu, std

        self._mu_res, self._std_res = _fit_stats(y_res)
        self._mu_tps, self._std_tps = _fit_stats(y_tps)

        y_res_std = (y_res - self._mu_res) / self._std_res
        y_tps_std = (y_tps - self._mu_tps) / self._std_tps

        # Standardized tps threshold
        self._lam_tps_std = (self.lambda_tps - self._mu_tps) / self._std_tps

        # Fit target task GPs
        self.gp_res.fit(X_np, y_res_std)
        self.gp_tps.fit(X_np, y_tps_std)

        # Enable meta-learning if valid source models and weights exist
        self._meta_enabled = (
            len(self.source_models) > 0 and
            self.source_weights is not None and
            len(self.source_weights) == len(self.source_models) and
            np.any(self.source_weights > 0)
        )
        if self._meta_enabled:
            w = np.clip(self.source_weights.astype(float), 0.0, None)
            s = w.sum()
            self.source_weights = (w / s) if s > 0 else np.ones_like(w) / len(w)

    # ---------------- Internal methods ----------------

    def _register_source_models(self):
        """Load top historical surrogate model(s) and associated similarity weights."""
        rec = self._get_record(model_name=self.model_name, hardware=self.hardware)
        model_path = f'surrogate_models/{rec["top_hardware"]}/{rec["top_model"]}.pth'
        model_paths = [model_path]

        models = []
        for p in model_paths:
            sm = SurrogateModel.load_model(p)
            assert sm.num_objectives == 2, f"source model {p} must have 2 objectives (tps, res)"
            models.append(sm)
        w = np.array([rec["similarity"]], dtype=float)
        w = np.clip(w, 0.0, None)
        self.source_models = models
        self.source_weights = (w / w.sum()) if w.sum() > 0 else np.ones_like(w) / len(w)
        self._meta_enabled = (len(self.X) >= 2)

    def _get_record(self, model_name, hardware, filepath="meta_features/records.jsonl"):
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
                    return {
                        "top_model": top.get("model_name"),
                        "top_hardware": top.get("hardware"),
                        "similarity": top.get("similarity"),
                    }

        return {"top_model": None, "top_hardware": None, "similarity": None}

    def _meta_predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Return (mu_res, sd_res, mu_tps, sd_tps) in standardized space.
        - Target GP provides (mu_t, sd_t)
        - Source models provide standardized means (no variance transfer)
        - Combine means using similarity weights and target proportion.
        """
        mu_res_t, sd_res_t = self.gp_res.predict(X, return_std=True)
        mu_tps_t, sd_tps_t = self.gp_tps.predict(X, return_std=True)

        if not self._meta_enabled:
            return mu_res_t, sd_res_t, mu_tps_t, sd_tps_t

        mu_res_src_all = []
        mu_tps_src_all = []
        X_list = X.tolist()

        for sm in self.source_models:
            preds = sm.predict(X_list)
            mean_res = np.array(preds[1][0]).reshape(-1)
            mean_tps = np.array(preds[0][0]).reshape(-1)
            mu_res_src_all.append((mean_res - self._mu_res) / self._std_res)
            mu_tps_src_all.append((mean_tps - self._mu_tps) / self._std_tps)

        W = self.source_weights.reshape(-1, 1)
        mu_res_src = np.sum(W * np.stack(mu_res_src_all, axis=0), axis=0)
        mu_tps_src = np.sum(W * np.stack(mu_tps_src_all, axis=0), axis=0)

        wt_target = 0.6  # Target model weight in mean fusion
        mu_res = wt_target * mu_res_t + (1 - wt_target) * mu_res_src
        mu_tps = wt_target * mu_tps_t + (1 - wt_target) * mu_tps_src

        return mu_res, sd_res_t, mu_tps, sd_tps_t

    def _compute_cei(self, X: np.ndarray) -> np.ndarray:
        """
        Constrained Expected Improvement (CEI):
          - Objective: minimize res
          - Constraint: tps ≥ lambda_tps
          - Computed in standardized space.
        """
        if len(self.y_res) == 0:
            return np.zeros(X.shape[0])

        mu_res, sd_res, mu_tps, sd_tps = self._meta_predict(X)

        y_res = np.array(self.y_res)
        y_tps = np.array(self.y_tps)
        feas_mask = (y_tps >= self.lambda_tps)
        f_best = float(np.min(y_res[feas_mask])) if np.any(feas_mask) else float(np.min(y_res))
        f_best_std = (f_best - self._mu_res) / self._std_res

        imp = f_best_std - mu_res
        with np.errstate(divide='ignore', invalid='ignore'):
            Z = imp / sd_res
            ei = imp * norm.cdf(Z) + sd_res * norm.pdf(Z)
            ei[sd_res == 0.0] = 0.0

        z_tps = (mu_tps - self._lam_tps_std) / sd_tps
        p_tps = norm.cdf(z_tps)

        cei = ei * p_tps
        return np.nan_to_num(cei, nan=0.0, posinf=0.0, neginf=0.0)

    def _encode_config(self, config):
        """Encode configuration into [0,1] normalized numeric vector for GP input."""
        encoded = []
        for name in self.param_names:
            val = config[name]
            pinfo = self.parameters[name]
            ptype = self.param_types[name]
            if ptype in ('integer', 'float'):
                vmin = pinfo['values']['min']; vmax = pinfo['values']['max']
                encoded.append(0.0 if vmax == vmin else (float(val) - vmin) / (vmax - vmin))
            elif ptype == 'enum':
                options = pinfo['values']
                encoded.append(0.0 if len(options) <= 1 else options.index(val) / (len(options) - 1))
            elif ptype == 'boolean':
                encoded.append(1.0 if val else 0.0)
        return np.array(encoded, dtype=float)


class ScootTuner(BaseTuner):
    """
    MOBO + EHVI tuner (2 objectives: maximize TPS, minimize RES) with:
      - RF-based hidden-constraint model (predicts feasibility POF in [0,1])
      - Dynamic feasibility threshold Δ
      - 2D EHVI (MC approximation) in the space (f1 = -tps, f2 = res)
    """

    def __init__(self,
                 parameters_path: str,
                 known_constraints: List[str],
                 objectives: List[str],
                 device: str,
                 hardware: str,
                 parallel_candidates: int = 5,
                 init_multiplier: float = 1.0,
                 delta_nu: float = 0.05,
                 pof_min: float = 0.25,
                 pof_max: float = 0.75,
                 random_state: int = 42):
        super().__init__(parameters_path, known_constraints, objectives, device, hardware)

        # Data buffers (original scale)
        self.X: List[np.ndarray] = []         # encoded feasible samples
        self.y_tps: List[float] = []          # TPS (larger is better)
        self.y_res: List[float] = []          # RES (smaller is better)
        self.feas_X: List[np.ndarray] = []    # for RF: feasible + infeasible samples
        self.feas_y: List[float] = []         # 1.0 feasible, 0.0 infeasible

        self.random_state = np.random.RandomState(random_state)
        self.n_dim = len(self.param_names)

        # Resource metric name (project convention)
        self.resource_metric = 'gpu_avg' if device == 'gpu' else 'mem_avg'
        self._fallback_resource_metrics = ['gpu_avg', 'mem_avg']

        # GPs (targets standardized internally)
        kernel = C(1.0, (1e-3, 1e3)) * Matern(nu=1.5) + W(1e-3, (1e-6, 1e-1))
        self.gp_tps = GaussianProcessRegressor(kernel=kernel, normalize_y=False, random_state=random_state)
        self.gp_res = GaussianProcessRegressor(kernel=kernel, normalize_y=False, random_state=random_state)
        self._mu_tps = 0.0; self._std_tps = 1.0
        self._mu_res = 0.0; self._std_res = 1.0

        # RF for feasibility (POF in [0,1])
        self.rf = RandomForestRegressor(
            n_estimators=200,
            max_depth=None,
            min_samples_leaf=2,
            random_state=random_state,
            n_jobs=-1
        )

        # Dynamic Δ (Algorithm 1 behavior)
        self.delta = 0.5
        self.delta_nu = float(delta_nu)
        self.delta_min = float(pof_min)
        self.delta_max = float(pof_max)
        self._consec_feasible_suggestions = 0
        self._init_done = False
        self._init_needed = max(1, int(math.ceil(self.n_dim * init_multiplier)))

        # Candidate batch size and reference point
        self.parallel_candidates = int(parallel_candidates)
        self._ref_point: Optional[np.ndarray] = None            # in (f1=-tps, f2=res)
        self._cached_enum_maps = self._build_enum_maps()

    # ---------------- Public API ----------------

    def suggest_configurations(self, k: int) -> List[Dict]:
        """
        Return k suggestions using 2D EHVI with constraint filtering.
        Init phase: uniform sampling (Sobol/QMC equivalent).
        """
        if not self._init_done or len(self.X) < self._init_needed:
            n = max(k, self._init_needed - len(self.X))
            return self._uniform_sample(n)

        raw_cands = self.executor.generate_configs(self.parameters, n_samples=self.parallel_candidates)

        # Known constraints
        cands = [c for c in raw_cands if self._satisfy_known_constraints(c)]
        if len(cands) == 0:
            cands = raw_cands

        # Hidden constraints via RF (keep POF >= Δ)
        if len(self.feas_X) >= 3:
            Xc = np.vstack([self._encode_config(cc) for cc in cands])
            pof = np.clip(self.rf.predict(Xc), 0.0, 1.0)
            keep = pof >= self.delta
            cands = [c for c, ok in zip(cands, keep) if ok]

        # Backfill if too few candidates
        if len(cands) < max(k, 16):
            backfill = self.executor.generate_configs(self.parameters, n_samples=self.parallel_candidates)
            backfill = [c for c in backfill if self._satisfy_known_constraints(c)]
            cands.extend(backfill)

        if len(cands) == 0:
            return self._uniform_sample(k)

        Xc_enc = np.vstack([self._encode_config(c) for c in cands])
        scores = self._ehvi_scores(Xc_enc, n_mc=256)
        top_idx = np.argsort(scores)[-k:][::-1]
        return [cands[i] for i in top_idx]

    def update(self, configs: List[Dict], performances: List[Optional[Dict]]):
        """
        Incorporate new observations, train RF each round, fit GPs when feasible data is sufficient,
        update Δ dynamically, and refresh the reference point for EHVI.
        """
        any_invalid = False

        for cfg, perf in zip(configs, performances):
            x_enc = self._encode_config(cfg)
            feasible, tps_val, res_val = self._parse_perf(perf)

            self.feas_X.append(x_enc)
            self.feas_y.append(1.0 if feasible else 0.0)

            if feasible:
                self.X.append(x_enc)
                self.y_tps.append(float(tps_val))
                self.y_res.append(float(res_val))
            else:
                any_invalid = True

        if not self._init_done and len(self.X) >= self._init_needed:
            self._init_done = True

        # Dynamic Δ
        if any_invalid:
            self.delta = min(self.delta_max, max(0.5, self.delta + self.delta_nu))
            self._consec_feasible_suggestions = 0
        else:
            self._consec_feasible_suggestions += len(configs)
            if self._consec_feasible_suggestions >= 5:
                self.delta = max(self.delta_min, self.delta - self.delta_nu)
                self._consec_feasible_suggestions -= 5

        # Train/refresh RF
        if len(self.feas_X) >= 3:
            Xrf = np.vstack(self.feas_X)
            yrf = np.asarray(self.feas_y, dtype=float)
            self.rf.fit(Xrf, yrf)

        # Fit GPs when enough feasible data
        if len(self.X) >= 3:
            X_np = np.vstack(self.X)
            y_tps = np.asarray(self.y_tps, dtype=float)
            y_res = np.asarray(self.y_res, dtype=float)

            self._mu_tps, self._std_tps = self._fit_stats(y_tps)
            self._mu_res, self._std_res = self._fit_stats(y_res)
            y_tps_std = (y_tps - self._mu_tps) / self._std_tps
            y_res_std = (y_res - self._mu_res) / self._std_res

            self.gp_tps.fit(X_np, y_tps_std)
            self.gp_res.fit(X_np, y_res_std)

            # Update reference point r in (f1=-tps, f2=res)
            f1_vals = -y_tps
            f2_vals = y_res
            r1 = float(np.max(f1_vals)) + 0.05 * (np.max(f1_vals) - np.min(f1_vals) + 1e-6)
            r2 = float(np.max(f2_vals)) + 0.05 * (np.max(f2_vals) - np.min(f2_vals) + 1e-6)
            self._ref_point = np.array([r1, r2], dtype=float)

    # ---------------- Internals ----------------

    def _uniform_sample(self, n: int) -> List[Dict]:
        """Uniform sampling via executor (Sobol/QMC-equivalent init)."""
        cands = self.executor.generate_configs_fixed(self.parameters, n_samples=n * 3)
        cands = [c for c in cands if self._satisfy_known_constraints(c)]
        if len(cands) < n:
            return cands
        uniq = {}
        for c in cands:
            key = tuple(self._encode_config(c).tolist())
            if key not in uniq:
                uniq[key] = c
            if len(uniq) >= n:
                break
        return list(uniq.values())[:n]

    def _satisfy_known_constraints(self, cfg: Dict) -> bool:
        """Validate known constraints using executor if available; otherwise evaluate simple boolean rules."""
        if hasattr(self.executor, "is_valid"):
            try:
                return bool(self.executor.is_valid(cfg, self.known_constraints))
            except Exception:
                pass
        local_vars = {k: cfg.get(k) for k in cfg.keys()}
        for rule in self.known_constraints or []:
            rule = rule.strip()
            if not rule:
                continue
            try:
                ok = eval(rule, {"__builtins__": {}}, local_vars)
                if not bool(ok):
                    return False
            except Exception:
                continue
        return True

    def _build_enum_maps(self) -> Dict[str, Dict]:
        """Precompute enum value -> index maps."""
        maps = {}
        for name, spec in self.parameters.items():
            if spec["type"] == "enum":
                maps[name] = {v: i for i, v in enumerate(spec["choices"])}
        return maps

    def _encode_config(self, cfg: Dict) -> np.ndarray:
        """Map discrete/bool/enum to numeric; keep continuous as float."""
        xs = []
        for name in self.param_names:
            v = cfg[name]
            t = self.param_types[name]
            if t in ("int", "integer", "float"):
                xs.append(float(v))
            elif t in ("bool", "boolean"):
                xs.append(1.0 if bool(v) else 0.0)
            elif t == "enum":
                idx = self._cached_enum_maps[name].get(v, 0)
                xs.append(float(idx))
            else:
                try:
                    xs.append(float(v))
                except Exception:
                    xs.append(0.0)
        return np.asarray(xs, dtype=float)

    @staticmethod
    def _fit_stats(v: np.ndarray) -> Tuple[float, float]:
        """Return mean and robust std (>=1e-8)."""
        mu = float(np.mean(v))
        sd = float(np.std(v))
        if sd < 1e-8:
            sd = 1.0
        return mu, sd

    def _gp_predict_stdspace(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        GP predictions in original target scale, then mapped to minimization space:
        returns (mu_f1, std_f1, mu_f2, std_f2) with f1 = -tps, f2 = res.
        """
        mu_tps_std, std_tps_std = self.gp_tps.predict(X, return_std=True)
        mu_tps = mu_tps_std * self._std_tps + self._mu_tps
        std_tps = np.maximum(std_tps_std * self._std_tps, 1e-9)

        mu_res_std, std_res_std = self.gp_res.predict(X, return_std=True)
        mu_res = mu_res_std * self._std_res + self._mu_res
        std_res = np.maximum(std_res_std * self._std_res, 1e-9)

        mu_f1 = -mu_tps
        std_f1 = std_tps
        mu_f2 = mu_res
        std_f2 = std_res
        return mu_f1, std_f1, mu_f2, std_f2

    # ---------- 2D EHVI (MC approximation) ----------

    def _ehvi_scores(self, Xc: np.ndarray, n_mc: int = 256) -> np.ndarray:
        """
        Compute EHVI for candidates Xc in 2D minimization space (f1=-tps, f2=res).
        Assumes independence between outputs.
        """
        if self._ref_point is None or len(self.X) < 2:
            return self.random_state.rand(len(Xc))

        Y = np.column_stack([-np.asarray(self.y_tps, dtype=float),
                             np.asarray(self.y_res, dtype=float)])
        pareto_Y = self._pareto_minimize(Y)
        base_hv = self._hv_2d(pareto_Y, self._ref_point)

        mu_f1, sd_f1, mu_f2, sd_f2 = self._gp_predict_stdspace(Xc)

        scores = np.zeros(len(Xc), dtype=float)
        for i in range(len(Xc)):
            if (not np.isfinite(sd_f1[i]) or not np.isfinite(sd_f2[i])
                or sd_f1[i] <= 0 or sd_f2[i] <= 0):
                y_new = np.array([mu_f1[i], mu_f2[i]])
                scores[i] = self._hv_inc_single(pareto_Y, self._ref_point, base_hv, y_new)
                continue

            s1 = self.random_state.normal(mu_f1[i], sd_f1[i], size=n_mc)
            s2 = self.random_state.normal(mu_f2[i], sd_f2[i], size=n_mc)
            imp = 0.0
            for a, b in zip(s1, s2):
                y_new = np.array([a, b], dtype=float)
                imp += self._hv_inc_single(pareto_Y, self._ref_point, base_hv, y_new)
            scores[i] = imp / float(n_mc)

        return scores

    @staticmethod
    def _pareto_minimize(Y: np.ndarray) -> np.ndarray:
        """Return non-dominated set for 2D minimization."""
        if len(Y) == 0:
            return Y
        keep = np.ones(len(Y), dtype=bool)
        for i in range(len(Y)):
            if not keep[i]:
                continue
            dominated = (np.all(Y <= Y[i], axis=1) & np.any(Y < Y[i], axis=1))
            dominated[i] = False
            if np.any(dominated):
                keep[i] = False
        return Y[keep]

    @staticmethod
    def _hv_2d(P: np.ndarray, r: np.ndarray) -> float:
        """2D minimization hypervolume w.r.t. reference point r."""
        if P.size == 0:
            return 0.0
        P = ScootTuner._pareto_minimize(P)
        P = P[np.argsort(P[:, 0])]
        hv = 0.0
        prev_f2 = r[1]
        for f1, f2 in P:
            if f1 >= r[0]:
                continue
            h = max(0.0, prev_f2 - min(f2, prev_f2))
            w = max(0.0, r[0] - f1)
            hv += w * h
            prev_f2 = min(prev_f2, f2)
        return hv

    @staticmethod
    def _hv_inc_single(P: np.ndarray, r: np.ndarray, base_hv: float, y_new: np.ndarray) -> float:
        """Hypervolume improvement from adding a single point in 2D minimization."""
        if np.any(y_new >= r):
            return 0.0
        P_new = np.vstack([P, y_new[None, :]])
        hv_new = ScootTuner._hv_2d(P_new, r)
        return max(0.0, hv_new - base_hv)

    # ---------- Parsing observations ----------

    def _parse_perf(self, perf: Optional[Dict]) -> Tuple[bool, Optional[float], Optional[float]]:
        """
        Return (feasible, tps, res).
        If perf is None or required keys are missing -> infeasible.
        """
        if perf is None:
            return False, None, None
        try:
            tps_val = float(perf['tps_avg'])
            res_val = float(perf[self.resource_metric])
        except Exception:
            return False, None, None
        return True, tps_val, res_val
