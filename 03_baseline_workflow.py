from hv_calculator import HypervolumeCalculator
import json
import argparse
import matplotlib.pyplot as plt
from typing import List, Dict, Any, Tuple
import numpy as np
from basetuner import DefaultTuner, RandomTuner, GeneticAlgorithmTuner, ConstrainedBayesTuner

class BaselineWorkflow:
    def __init__(self, 
                 parameters_path: str, 
                 known_constraints: List[str],
                 objectives: List[str],
                 algorithm: str = "RD", 
                 max_observations: int = 30,
                 parallel_degree: int = 5,
                 device: str = "gpu",
                 hardware: str = "m4",
                 model: str = "qwen3-8b",
                 quant: str = "q4"):
        
        # 根据算法选择调优器
        self.method = algorithm
        if algorithm == "Default":
            self.tuner = DefaultTuner(parameters_path, known_constraints, objectives, device=device, hardware=hardware)
        elif algorithm == "RD":
            self.tuner = RandomTuner(parameters_path, known_constraints, objectives, device=device, hardware=hardware)
        elif algorithm == "GA":
            self.tuner = GeneticAlgorithmTuner(parameters_path, known_constraints, objectives, device=device, hardware=hardware)
        elif algorithm == "CBO":
            self.tuner = ConstrainedBayesTuner(parameters_path, known_constraints, objectives, device=device, hardware=hardware)
        else:
            raise ValueError(f"Unknown algorithm: {algorithm}")
        
        self.max_observations = max_observations
        self.hardware = hardware
        self.model_name = f"{model}-{quant}"
        self.model = f"./../models/{model}-{quant}.gguf"
        self.parallel_degree = parallel_degree
        self.current_observations = 0
        self.objectives = objectives
        self.history = {'configs': [], 'performance': []}
        self.iter_hv = []
        self.bounds = self._load_metric_bounds(f"bounds/{self.hardware}/{self.model_name}.json")
        self.hv_calc = HypervolumeCalculator(self.bounds)
        self.pareto_front = []

    def run_workflow(self):
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
            
            print(f"Iteration {self.current_observations//self.parallel_degree + 1} "
                f"Observations: {self.current_observations}/{self.max_observations}")
            
        # self._plot_hv_over_iterations()

    def _load_metric_bounds(self, path: str) -> Dict[str, Tuple[float, float]]:
        """
        从 JSON 文件读取 {"metric": {"min": x, "max": y}, ...}
        返回 {"metric": (min, max)}，数值会转成 float。
        无效/缺失的项会被跳过。
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
        perform = self._evaluate_configs([{}])
        self.update_pareto_front()
        hv = self._compute_hypervolume()
        self.iter_hv.append(hv)

    def _evaluate_configs(self, configs: List[Dict], init_model=False):
        """评估配置"""
        perf_results = []
        
        for config in configs:
            # 模拟评估过程
            try:
                print(config)
                result = self.tuner.executor.run_server_performance_test(config, model_path=self.model)
                # perf = {"tps_avg": result['tps_avg'], "gpu_avg": result['gpu_avg'], "mem_avg": result['mem_avg']}
                perf = {metric: result[metric] for metric in self.objectives}
                # print(perf)

            except ValueError as e:
                # perf = {metric: None for metric in self.objectives}
                continue

            print(f"perf: {perf}")

            # 记录结果
            self.history['configs'].append(config)
            self.history['performance'].append(perf)
            self.current_observations += 1
            
            perf_results.append(perf)
            
            # 提前停止检查
            if self.current_observations >= self.max_observations:
                break

        # if init_model:
        #     self.tuner.update_surrogate()
        #     print("==SURROGATE MODEL UPDATED==")
                
        return perf_results
    
    def update_pareto_front(self):
        """更新Pareto前沿（每次新观测后调用）"""
        new_front = []
        # 筛选可行解
        # feasible_solutions = #todo
        
        # 支配关系计算
        for candidate in zip(self.history['configs'], self.history['performance']):
            dominated = False
            # 检查是否被现有前沿支配
            for front_sol in new_front:
                if self._dominates(front_sol[1], candidate[1]):
                    dominated = True
                    break
            if not dominated:
                # 移除被新解支配的旧解
                new_front = [sol for sol in new_front 
                           if not self._dominates(candidate[1], sol[1])]
                new_front.append(candidate)
        
        self.pareto_front = new_front

    def _dominates(self, a: Dict[str, float], b: Dict[str, float]) -> bool:
        """判断a是否支配b"""
        better_in_any = False
        for obj, direction in self.objectives.items():
            if direction == 'min':
                if a[obj] > b[obj]: return False
                if a[obj] < b[obj]: better_in_any = True
            else:
                if a[obj] < b[obj]: return False
                if a[obj] > b[obj]: better_in_any = True
        return better_in_any

    def _compute_hypervolume(self) -> float:
        """计算当前 Pareto front 的归一化超体积 (tps_avg↑, gpu_avg↓)"""
        if not self.history['performance']:
            return 0.0

        pareto_front = [t[1] for t in self.pareto_front]
        if not pareto_front:
            return 0.0

        hv = self.hv_calc.compute(pareto_front)
        print(f"Current Hypervolume: {hv:.4f}")

        return hv

    def _convert_to_serializable(self, obj: Any) -> Any:
        """将 numpy 类型转换为原生 Python 类型"""
        if isinstance(obj, (np.integer, )):
            return int(obj)
        elif isinstance(obj, (np.floating, )):
            return float(obj)
        elif isinstance(obj, (np.bool_, )):
            return bool(obj)
        elif isinstance(obj, dict):
            return {k: self._convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_to_serializable(i) for i in obj]
        else:
            return obj

    def get_best_config(self) -> Dict:
        """获取当前最优配置"""
        # 根据目标类型选择最佳配置
        if len(self.tuner.objectives) == 1:
            return self._get_single_best()
        else:
            return self._get_mul_best()

    def _get_single_best(self) -> Dict:
        """单目标最优"""
        obj = list(self.objectives.keys())[0] 
        direct = list(self.objectives.values())[0] 
        #perfs = self.history['performance'][obj]
        perfs = [perf[obj] for perf in self.history['performance']]
        best_idx = np.argmin(perfs) if direct == 'min' else np.argmax(perfs)
        print(self.history['performance'][best_idx])
        return self.history['configs'][best_idx]
    
    def _get_mul_best(self) -> Dict:
        """多目标最优"""
        # 计算每个配置的综合得分
        perfs = self.history['performance']
        best_idx, best_score = self._evaluate_multi_obj(perfs)
        print(best_score)
        print(perfs[best_idx])
        return self.history['configs'][best_idx]
    
    def _evaluate_multi_obj(self, perfs):
        tps = np.array([perf['tps_avg'] for perf in perfs])
        gpu = np.array([perf['gpu_avg'] for perf in perfs])
        # mem = np.array([perf['mem_avg'] for perf in perfs])

        # 归一化函数
        def normalize(arr, is_benefit=True):
            """将指标归一化到[0,1]范围
            is_benefit=True表示指标越大越好，False表示越小越好"""
            if is_benefit:
                return (arr - np.min(arr)) / (np.max(arr) - np.min(arr))
            else:
                return (np.max(arr) - arr) / (np.max(arr) - np.min(arr))

        # 归一化处理
        tps_norm = normalize(tps, is_benefit=True)  # tps越大越好
        gpu_norm = normalize(gpu, is_benefit=False)  # resource越小越好
        # mem_norm = normalize(mem, is_benefit=False)  # mem越小越好

        # 设置权重
        w_tps = 0.7
        w_gpu = 0.3
        # w_mem = 0.2

        # 计算综合得分
        scores = w_tps * tps_norm + w_gpu * gpu_norm

        # 找到最优解
        best_idx = np.argmax(scores)
        best_score = scores[best_idx]

        return best_idx, best_score
    
    def save_pareto_front_and_hv(self, method:str, model: str):
        """保存 Pareto 前沿到指定文件"""
        if self.objectives == 1:
            print("当前仅支持多目标 Pareto 前沿保存。")
            return
        
        if not hasattr(self, 'pareto_front'):
            print("当前不存在 Pareto 前沿，无法保存。")
            return

        pareto_serializable = [
            {
                "config": self._convert_to_serializable(config),
                "perf": self._convert_to_serializable(perf)
            }
            for config, perf in self.pareto_front
        ]

        with open(f"pareto_fronts/{self.hardware}/{method}-{model}.json", 'w', encoding='utf-8') as f:
            json.dump(pareto_serializable, f, indent=2)
        print(f"Pareto 前沿已保存到 pareto_fronts/{self.hardware}/{method}-{model}.json")

        with open(f"hv_progress/{self.hardware}/{method}-{model}.json", "w") as f:
            json.dump(self.iter_hv, f, indent=4)
        print(f"save to hv_progress/{self.hardware}/{method}-{model}.json")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Llama Configuration Optimizer')
    parser.add_argument('--device', type=str, choices=['cpu', 'gpu'], default='gpu',
                       help='Processing device (cpu or gpu)')
    parser.add_argument('--hardware', type=str, choices=['rtx3060', 'rtx4090', 'm4', 'orin'], default='rtx3060',
                       help='Processing hardware')
    parser.add_argument('--method', type=str, choices=['Default', 'GA', 'SCOOT', 'CBO'], default='GA',
                       help='Processing mothod (GA, SCOOT, CBO)')
    parser.add_argument('--model', type=str, choices=['qwen3-4b','phimoe-mini'], default='phimoe-mini',
                        help='qwen3-8b, phimoe-mini')
    parser.add_argument('--quant', type=str, choices=['q4','q8'],default='q8',
                        help='q4, q8')
    args = parser.parse_args()

    if args.device == 'gpu':
        objectives = {'tps_avg': 'max', 'gpu_avg': 'min'}
    else:
        objectives = {'tps_avg': 'max', 'mem_avg': 'min'}

    workflow = BaselineWorkflow(
        parameters_path=f"knobs_files/{args.hardware}/{args.model}-{args.quant}.json",
        known_constraints=[],
        objectives=objectives,
        algorithm=args.method, 
        max_observations=50,
        parallel_degree=5,
        device=args.device,
        hardware=args.hardware,
        model = args.model,
        quant = args.quant
    )
    workflow.run_workflow()
    workflow.save_pareto_front_and_hv(method=args.method, model=f"{args.model}-{args.quant}")
