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
    def __init__(self, 
                 parameters_path: str, 
                 objectives: Dict,
                 max_observations: int = 30,
                 parallel_degree: int = 5,
                 device: str = "gpu",
                 hardware: str = "rtx3060",
                 model: str = "qwen3-4b",
                 quant: str = "q4"):
        """
        :param parameters: 参数空间定义
        :param objectives: 优化目标列表
        :param max_observations: 最大观察次数阈值
        :param parallel_degree: 并行度
        """
        # 初始化核心组件
        self.tuner = LaTune(parameters_path, objectives)
        self.objectives = objectives
        self.max_observations = max_observations
        self.parallel_degree = parallel_degree
        self.executor = LlamaExecutor(self.tuner.param_types, device=device,hardware=hardware)
        self.hardware = hardware
        self.model_name = f"{model}-{quant}"
        self.model = f"./../models/{model}-{quant}.gguf"
        
        # 工作流程状态跟踪
        self.current_observations = 0
        self.history = {
            'configs': [],
            'performance': []
        }
        self.iter_hv = []
        self.bounds = self._load_metric_bounds(f"bounds/{self.hardware}/{self.model_name}.json")
        self.hv_calc = HypervolumeCalculator(self.bounds)
        
    def run_workflow(self):
        """执行完整的工作流程"""
        # 步骤1-2: 初始采样和评估
        initial_samples = self._generate_initial_samples()
        self._evaluate_configs(initial_samples,init_model=True)
        
        # 迭代调优（步骤3-9）
        while self.current_observations < self.max_observations:
            start_time = time.time()
            
            # 步骤3: 建议新配置
            suggested_configs = self.tuner.suggest_configurations(k=self.parallel_degree)
            
            # 步骤4: 评估
            new_perfs = self._evaluate_configs(suggested_configs)
                        

            # 步骤7: 更新代理模型
            self.tuner.update_surrogate(len(new_perfs))

            self.tuner.update_pareto_front()  # 更新Pareto前沿

            self.tuner.set_reference_point()
            
            # 记录迭代信息
            iteration_time = time.time() - start_time
            print(f"Iteration {self.tuner.iteration} completed. "
                  f"Observations: {self.current_observations}/{self.max_observations} "
                  f"Time: {iteration_time:.1f}s")

            self.tuner.iteration = self.tuner.iteration + 1

            hv = self._compute_hypervolume()
            self.iter_hv.append(hv)

            if self.current_observations >= self.max_observations:
                self.tuner.save_surrogate(f"surrogate_models/{self.hardware}/{self.model_name}.pth")
                # self._plot_hv_over_iterations()
                break

    def _generate_initial_samples(self) -> List[Dict]:
        """生成初始样本（步骤1）"""
        return self.tuner.generate_initial_samples(5)
    
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

    
    def _evaluate_configs(self, configs: List[Dict], init_model=False):
        """评估配置"""
        perf_results = []
        
        for config in configs:
            # 模拟评估过程
            try:
                result = self.executor.run_server_performance_test(config, model_path=self.model)
                # perf = {"tps_avg": result['tps_avg'], "gpu_avg": result['gpu_avg'], "mem_avg": result['mem_avg']}
                perf = {metric: result[metric] for metric in self.objectives}
                # print(perf)

            except ValueError as e:
                # perf = {metric: None for metric in self.objectives}
                continue

            self.tuner.observations.append((config, perf))
            print(f"config: {config}, perf: {perf}")

            # 记录结果
            self.history['configs'].append(config)
            self.history['performance'].append(perf)
            self.current_observations += 1
            
            perf_results.append(perf)
            
            # 提前停止检查
            if self.current_observations >= self.max_observations:
                break

        if init_model:
            self.tuner.update_surrogate()

            self.tuner.update_pareto_front()  # 更新Pareto前沿

            self.tuner.set_reference_point()

            hv = self._compute_hypervolume()
            self.iter_hv.append(hv)
            print("==SURROGATE MODEL UPDATED==")
                
        return perf_results

    def get_best_config(self) -> Dict:
        """获取当前最优配置"""
        # 根据目标类型选择最佳配置
        if len(self.tuner.objectives) == 1:
            return self._get_single_best()
        else:
            return self._get_pareto_best()

    def _get_single_best(self) -> Dict:
        """单目标最优"""        
        obj = list(self.objectives.keys())[0] 
        direct = list(self.objectives.values())[0] 
        #perfs = self.history['performance'][obj]
        perfs = [perf[obj] for perf in self.history['performance']]
        best_idx = np.argmin(perfs) if direct == 'min' else np.argmax(perfs)
        print(self.history['performance'][best_idx])
        return self.history['configs'][best_idx]

    def _get_pareto_best(self) -> List[Dict]:
        """获取Pareto前沿的最优解"""
        pareto_front =  self.tuner.get_pareto_front()
        print(pareto_front)
        perfs = [sample[1] for sample in pareto_front]
        best_idx, best_score = self.tuner.evaluate_pareto(perfs)
        print(best_score)
        print(pareto_front[best_idx])
        return pareto_front[best_idx][0]

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
        
    def _compute_hypervolume(self) -> float:
        """计算当前 Pareto front 的归一化超体积 (tps_avg↑, gpu_avg↓)"""
        if not self.history['performance']:
            return 0.0

        pareto_front = [t[1] for t in self.tuner.get_pareto_front()]
        if not pareto_front:
            return 0.0

        hv = self.hv_calc.compute(pareto_front)
        print(f"Current Hypervolume: {hv:.4f}")

        return hv


    def _plot_hv_over_iterations(self):
        """绘制迭代过程中最优点（HV）的变化折线图"""
        plt.figure(figsize=(8, 5))
        plt.plot(range(1, len(self.iter_hv) + 1), self.iter_hv, marker='o')
        plt.xlabel("Iteration")
        plt.ylabel("Hypervolume")
        plt.title("Best Hypervolume Over Iterations")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig("hypervolume_progress_latune.png")
        plt.show()

    def save_pareto_front_and_hv(self, model: str):
        """保存 Pareto 前沿到指定文件"""
        if self.objectives == 1:
            print("当前仅支持多目标 Pareto 前沿保存。")
            return
        
        if not hasattr(self.tuner, 'pareto_front'):
            print("当前不存在 Pareto 前沿，无法保存。")
            return

        pareto_serializable = [
            {
                "config": self._convert_to_serializable(config),
                "perf": self._convert_to_serializable(perf)
            }
            for config, perf in self.tuner.pareto_front
        ]

        with open(f"pareto_fronts/{self.hardware}/{model}-latune.json", 'w', encoding='utf-8') as f:
            json.dump(pareto_serializable, f, indent=2)
        print(f"Pareto 前沿已保存到 pareto_fronts/{self.hardware}/{model}-latune.json")

        with open(f"hv_progress/{self.hardware}/{model}-latune.json", "w") as f:
            json.dump(self.iter_hv, f, indent=4)
        print(f"save to hv_progress/{self.hardware}/{model}-latune.json")

    def load_pareto_front(self, filepath: str):
        """从文件中加载 Pareto 前沿"""
        filepath = Path(filepath)
        if not filepath.exists():
            print(f"文件 {filepath} 不存在。")
            return

        with open(filepath, 'r', encoding='utf-8') as f:
            pareto_data = json.load(f)

        self.tuner.pareto_front = [
            (item["config"], item["perf"]) for item in pareto_data
        ]
        print(f"Pareto 前沿已从 {filepath} 加载。")

    def visualize_pareto_front(self):
        """Pareto前沿可视化（以双目标为例）"""
        import matplotlib.pyplot as plt
        
        front = self.tuner.get_pareto_front()
        print(front)
        perfs = [sample[1] for sample in front]
        if len(self.tuner.objectives) != 2:
            print("可视化仅支持双目标")
            return
        
        # obj_names = list(self.tuner.objectives.keys())
        # x = [p[obj_names[0]] for p in front]
        # y = [p[obj_names[1]] for p in front]
        y = np.array([perf['tps_avg'] for perf in perfs])
        x = np.array([perf['gpu_avg'] for perf in perfs])
        
        plt.scatter(x, y, c='red', label='Pareto Front')
        plt.xlabel('gpu_avg')
        plt.ylabel('tps_avg')
        plt.title("Pareto Front Evolution")
        plt.legend()
        plt.show()
        # 保存成PDF
        plt.savefig("tps_gpu_r3.pdf")

# 使用示例
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Llama Configuration Optimizer')
    parser.add_argument('--device', type=str, choices=['cpu', 'gpu'], default='gpu',
                       help='Processing device (cpu or gpu)')
    parser.add_argument('--hardware', type=str, choices=['rtx3060', 'rtx4090', 'm4', 'orin'], default='rtx3060',
                       help='Processing hardware')
    parser.add_argument('--model', type=str, choices=['qwen3-4b','phimoe-mini'], default='phimoe-mini',
                        help='qwen3-8b, phimoe-mini')
    parser.add_argument('--quant', type=str, choices=['q4','q8'],default='q8',
                        help='q4, q8')
    args = parser.parse_args()
    parameters_path = f"knobs_files/{args.hardware}/{args.model}-{args.quant}.json"

    if args.device == 'gpu':
        objectives = {'tps_avg': 'max', 'gpu_avg': 'min'}
    else:
        objectives = {'tps_avg': 'max', 'mem_avg': 'min'}
    
    # 初始化工作流程
    workflow = TUNINGWorkflow(
        parameters_path=parameters_path,
        objectives=objectives, 
        max_observations=50,
        parallel_degree=5,
        device=args.device,
        hardware=args.hardware,
        model = args.model,
        quant = args.quant
    )
    
    # 运行完整工作流程
    workflow.run_workflow()
    
    # 输出结果
    print("\n=== Tuning Results ===")
    print(f"Total evaluations: {len(workflow.history['configs'])}")
    print(f"Best configuration:")
    # print(workflow.get_best_config())

    workflow.save_pareto_front_and_hv(f"{args.model}-{args.quant}")
