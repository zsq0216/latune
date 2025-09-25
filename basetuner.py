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
    def __init__(self, parameters_path: str, known_constraints: List[str], objectives: List[str], device: str, hardware: str):
        self.parameters = Knobs(parameters_path, 5, random= False).knobs
        self.objectives = objectives
        self.known_constraints = known_constraints
        self.param_types = {name: param["type"] for name, param in self.parameters.items()}
        self.device = device
        print(self.device)
        self.executor = LlamaExecutor(self.param_types, device = self.device, hardware=hardware)

    def _load_parameters(self, path: str) -> List[Dict]:
        # 加载参数定义（示例实现）
        with open(path, 'r') as f:
            return json.load(f)

    @abstractmethod
    def suggest_configurations(self, k: int) -> List[Dict]:
        """生成k个待评估的配置"""
        pass

    @abstractmethod
    def update(self, configs: List[Dict], performances: List[float]):
        """用新数据更新调优器内部状态"""
        pass

class DefaultTuner(BaseTuner):
    def suggest_configurations(self, k: int) -> List[Dict]:
        pass

    def update(self, configs: List[Dict], performances: List[float]):
        pass

class RandomTuner(BaseTuner):
    def suggest_configurations(self, k: int) -> List[Dict]:
        # 随机采样
        return self.executor.generate_configs_fixed(self.parameters ,n_samples=k)

    def update(self, configs: List[Dict], performances: List[float]):
        # 随机采样无需更新内部状态
        pass

class GeneticAlgorithmTuner(BaseTuner):
    def __init__(self, parameters_path: str, known_constraints: List[str], objectives: List[str], device: str,
                 hardware:str, population_size: int = 5, mutation_rate: float = 0.1):
        super().__init__(parameters_path, known_constraints, objectives, device, hardware)
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.population = self._initialize_population()
        self.fitness = None  # 延迟初始化
        self.device = device

    def _initialize_population(self) -> List[Dict]:
        return self.executor.generate_configs_fixed(self.parameters, self.population_size)

    def initialize_with_performance(self, performances: List[Dict]):
        # 只用于初始评估之后的一次性调用
        valid_perf = [perf['tps_avg'] for perf in performances if perf is not None]
        if len(valid_perf) != len(self.population):
            raise ValueError("初始化 performance 数量与 population 数量不一致")
        self.fitness = valid_perf

    def suggest_configurations(self, k: int) -> List[Dict]:
        if self.fitness is None:
            raise RuntimeError("尚未初始化适应度，请先调用 initialize_with_performance")
        parents = self._tournament_selection(k)
        offspring = self._crossover_and_mutate(parents)
        return offspring

    def _tournament_selection(self, k: int) -> List[Dict]:
        selected = []
        for _ in range(k):
            candidates = np.random.choice(len(self.population), size=2, replace=False)
            best_idx = candidates[np.argmax([self.fitness[i] for i in candidates])]
            selected.append(self.population[best_idx])
        return selected

    def _crossover_and_mutate(self, parents: List[Dict]) -> List[Dict]:
        offspring = []
        for i in range(0, len(parents), 1):
            p1 = parents[i]
            p2 = parents[i+1] if i+1 < len(parents) else p1
            child = {}
            for name, param in self.parameters.items():
                child[name] = p1[name] if np.random.rand() > 0.5 else p2[name]
                if np.random.rand() < self.mutation_rate:
                    child[name] = self._mutate_param(param)
            child = self.executor.handle_dependency(child)
            offspring.append(child)
        return offspring

    def _mutate_param(self, param_config: Dict) -> Any:
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
        # 正常代际进化过程中的更新
        perf_values = [perf['tps_avg'] for perf in performances if perf is not None]
        if len(perf_values) != len(configs):
            raise ValueError("performance 与 configs 数量不匹配")
        self.population.extend(configs)
        self.fitness.extend(perf_values)
        top_indices = np.argsort(self.fitness)[::-1][:self.population_size]
        self.population = [self.population[i] for i in top_indices]
        self.fitness = [self.fitness[i] for i in top_indices]

class ConstrainedBayesTuner(BaseTuner):
    def __init__(self, parameters_path, known_constraints, objectives, device, lambda_tps=97, lambda_pps=300):
        super().__init__(parameters_path, known_constraints, objectives, device)
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
        if len(self.X) < 5:
            return self.executor.generate_configs(self.parameters ,n_samples=k)

        candidates = self.executor.generate_configs(self.parameters, n_samples=100)
        X_candidates = np.array([list(config.values()) for config in candidates])
        cei_scores = self._compute_cei(X_candidates)

        top_indices = np.argsort(cei_scores)[-k:][::-1]
        selected_configs = [candidates[i] for i in top_indices]
        return selected_configs

    def update(self, configs: List[Dict], performances: List[Dict]):
        for config, perf in zip(configs, performances):
            if perf is None:
                continue
            self.X.append(list(config.values()))
            self.y_res.append(perf[self.resource_metric])  # 或其他资源指标
            self.y_tps.append(perf['tps_avg'])
            # self.y_pps.append(perf['pps_avg'])

        self.gp_res.fit(self.X, self.y_res)
        self.gp_tps.fit(self.X, self.y_tps)
        # self.gp_pps.fit(self.X, self.y_pps)

    def _compute_cei(self, X: np.ndarray) -> np.ndarray:
        mu_res, sigma_res = self.gp_res.predict(X, return_std=True)
        mu_tps, sigma_tps = self.gp_tps.predict(X, return_std=True)
        # mu_pps, sigma_pps = self.gp_pps.predict(X, return_std=True)

        feasible_mask = (np.array(self.y_tps) >= self.lambda_tps)# & (np.array(self.y_pps) <= self.lambda_pps)
        f_best = np.min(np.array(self.y_res)[feasible_mask]) if any(feasible_mask) else np.min(self.y_res)

        imp = f_best - mu_res
        Z = imp / sigma_res
        ei = imp * norm.cdf(Z) + sigma_res * norm.pdf(Z)
        ei[sigma_res == 0.0] = 0.0

        p_tps = 1.0 - norm.cdf((self.lambda_tps - mu_tps) / sigma_tps)
        # p_pps = 1.0 - norm.cdf((self.lambda_pps - mu_pps) / sigma_pps)

        cei = ei * p_tps #* p_pps
        return cei



