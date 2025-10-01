import time
from typing import List, Dict, Tuple, Any
import numpy as np
from llama_executor import LlamaExecutor
from knobs import Knobs
import json
from pathlib import Path
import signal
from threading import Thread, Event
from workload_simulator import SystemLoadGenerator 
import sys
import subprocess

class ConfigEvaluator:
    def __init__(self, 
                 parameters_path: str, 
                 pareto_front_path: str,
                 device: str = "gpu",
                 model_path: str = '',
                 parameter_num = 5):
        """
        :param parameters: 参数空间定义
        """
        # 初始化核心组件
        self.parameters = Knobs(parameters_path, parameter_num, random = False).knobs
        self.param_types = {name: params['type'] for name, params in self.parameters.items()}
        self.device = device
        self.executor = LlamaExecutor(self.param_types, model_path=model_path, device=device)
        self.model_path = model_path
        self.fluctuation_detector = SystemLoadGenerator()
        self.fluctuation_event = Event()
        self.performance_log = []
        self.start_time = time.time()
        self.stream_thread = None
        self._load_pareto_front(pareto_front_path)
        
    def _load_pareto_front(self, filepath: str):
        """从文件中加载 Pareto 前沿"""
        filepath = Path(filepath)
        if not filepath.exists():
            print(f"文件 {filepath} 不存在。")
            return

        with open(filepath, 'r', encoding='utf-8') as f:
            pareto_data = json.load(f)

        self.pareto_front = [
            (item["config"], item["perf"]) for item in pareto_data
        ]
        print(f"Pareto 前沿已从 {filepath} 加载。")

    def _handle_exit(self, sig, frame):
        print("\n🛑 收到中断信号，正在清理...")

        self._terminate_all()
        with open('performance_log.json', 'w') as f:
            json.dump(self.performance_log, f, indent=4)
        sys.exit(0)

    def _get_pareto_best(self, budget, ablation = False) -> Dict:
        """获取Pareto前沿的最优解"""
        # 构造 (config, perf) 只保留满足资源预算的项
        if self.device == 'gpu':
            valid_pareto = [
                (cfg, perf) for cfg, perf in self.pareto_front
                if perf['gpu_avg'] <= budget['gpu_avail']
                # and perf['mem_avg'] <= budget['mem_avail']  # 如果需要可取消注释
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
        """获取Pareto前沿的最优解"""
        perfs = [sample[1] for sample in self.pareto_front]
        best_idx, best_score = self._evaluate_pareto(perfs)
        return self.pareto_front[best_idx][0]

    def _evaluate_pareto(self, perfs, ablation = False) -> Tuple[int, float]:
        tps = np.array([perf['tps_avg'] for perf in perfs])
        # pps = np.array([perf['pps_avg'] for perf in perfs])
        # mem = np.array([perf['mem_avg'] for perf in perfs])
        if self.device == 'gpu':
            gpu = np.array([perf['gpu_avg'] for perf in perfs])

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
        # pps_norm = normalize(pps, is_benefit=True)  # pps越大越好
        # mem_norm = normalize(mem, is_benefit=False)  # mem越小越好
        if self.device == 'gpu':
            gpu_norm = normalize(gpu, is_benefit=False)  # resource越小越好

        # 设置权重
        w_tps = 0.7
        w_gpu = 0.3


        # 计算综合得分
        # if self.device == 'cpu':
        #     scores = w_tps * tps_norm + w_pps * pps_norm + w_mem * mem_norm
        # if self.device == 'gpu':
        scores = w_tps * tps_norm + w_gpu * gpu_norm

        if ablation:
            scores = tps_norm

        # 找到最优解
        best_idx = np.argmax(scores)
        best_score = scores[best_idx]

        return best_idx, best_score

    def run_monitoring_cycle(self, duration=50, loop = False):
        """
        通用的监控周期：选最优配置 → 启动采样 → 等待 → 清理
        """
        self.fluctuation_event.clear()
        self.executor.stop_event = Event()

        # Step 1: 获取预算与最优解
        budget = self.fluctuation_detector.get_available_resources()
        print(f"📊 可用资源预算: {budget}")
        best_config = self._get_pareto_best(budget)
        print(f"🚀 使用最佳配置启动服务: {best_config}")

        # Step 2: 启动采样线程
        self.stream_thread = Thread(
            target=self._start_streaming,
            args=(best_config, self.model_path),
            daemon=True
        )
        self.stream_thread.start()

        # Step 3: 等待
        if loop:
            while not self.fluctuation_event.is_set():
                time.sleep(2)
        else:
            time.sleep(duration)

        # Step 4: 终止采样线程与服务
        self._terminate_all()

    def adjust_monitoring(self, duration=50):
        print("\n🧪 手动调节监控中...")
        self.run_monitoring_cycle(duration=duration)

    def adaptive_monitoring_loop(self):
        print("\n🔁 自适应监控循环已启动...")
        signal.signal(signal.SIGINT, self._handle_exit)
        
        while True:
            self._start_fluctuation_monitor()
            self.run_monitoring_cycle(duration=50, loop=True)

    def fixed_monitoring(self, duration=50):
        best_config = self._get_pareto_best_no_budget()
        self.stream_thread = Thread(
            target=self._start_streaming,
            args=(best_config, self.model_path),
            daemon=True
        )
        self.stream_thread.start()

        time.sleep(duration)

        # Step 4: 终止采样线程与服务
        self._terminate_all()

    def evaluate_instance(self, method = 'latune', model = 'qwen3-4b-q4'):
        model = f"./../models/{model}.gguf"
        if method not in ["Default", "GA", "CBO", "scoot", "latune"]:
            raise ValueError("Invalid method. Choose from [Default, GA, CBO, scoot, latune]")
        if method == 'Default':
            return self.executor.run_server_performance_test(config={},model_path = model)
        budget = self.fluctuation_detector.get_available_resources()
        print(f"📊 可用资源预算: {budget}")
        best_config = self._get_pareto_best(budget)
        print(f"🚀 使用最佳配置启动服务: {best_config}")

        return self.executor.run_server_performance_test(config = best_config, model_path = model)

    def ablation_2(self, model_size = '4b'):
        model = f"./../../models/qwen3-{model_size}-q4.gguf"
        best_config = self._get_pareto_best_no_budget()
        print(f"🚀 使用最佳配置启动服务: {best_config}")
        return self.executor.run_server_performance_test(config = best_config, model_path = model)
    
    def ablation_3(self, model_size = '4b'):
        model = f"./../../models/qwen3-{model_size}-q4.gguf"
        budget = self.fluctuation_detector.get_available_resources()
        print(f"📊 可用资源预算: {budget}")
        best_config = self._get_pareto_best(budget, ablation=True)
        print(f"🚀 使用最佳配置启动服务: {best_config}")
        return self.executor.run_server_performance_test(config = best_config, model_path = model)    

    def _start_streaming(self, config, model_path):
        for report in self.executor.run_server_performance_test_streaming(config, model_path):
            print("📊 收集到指标:", report)
            timestamp = time.time() - self.start_time
            self.performance_log.append({"perf":report, "timestamp": timestamp})
            if self.fluctuation_event.is_set():
                break
            if self.executor.stop_event.is_set():
                break

    def _start_fluctuation_monitor(self):
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
        print("🧹 正在终止当前服务与线程...")
        self.executor.stop_event.set()
        self.stream_thread.join(timeout=3)
        self.fluctuation_detector.stop_event.set() 
        self.fluctuation_event.clear()
        if hasattr(self.executor, 'stop_event'):
            print("🧹 终止采样线程...")
            self.executor.stop_event.set()
        if hasattr(self.executor, 'server_process'):
            print("🧹 终止服务器进程...")
            self.executor.server_process.terminate()
            try:
                self.executor.server_process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.executor.server_process.kill()

if __name__ == "__main__":
    # 示例用法
    config_evaluator = ConfigEvaluator(
        parameters_path="ranked_knobs_files/rank_server_knobs_gpu.json",
        pareto_front_path="pareto_fronts/pareto_front_GA.json",
        device="gpu",
        model_path = './../../models/qwen2.5-1.5b-instruct-fp16.gguf'
    )
    
    # config_evaluator.adaptive_monitoring_loop()