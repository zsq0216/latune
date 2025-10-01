import numpy as np
from sklearn.ensemble import RandomForestClassifier
from scipy.stats import norm
from sobol_seq import i4_sobol_generate  # 需要安装sobol_seq库
from knobs import Knobs
from surrogate_model import SurrogateModel
from pymoo.indicators.hv import HV  # 需安装pymoo库
from typing import List, Dict, Tuple
import random
import json

class LaTune:
    def __init__(self, parameters_path,  objectives, delta_init=0.5, num_gpus=4):
        self.parameters = Knobs(parameters_path, 5, random = False).knobs   # 这样可以吗？
        self.delta = delta_init
        self.consecutive_feasible = 0
        self.rf = RandomForestClassifier(n_estimators=100)
        self.X = []          # 存储配置特征（编码为数值）
        self.y = []          # 存储可行性标签（0/1）
        self.observations = []  # 存储配置及其目标值
        self.iteration = 1
        self.num_gpus = num_gpus

        # 参数编码元数据
        self.param_names = list(self.parameters.keys())
        self.param_types = {name: params['type'] for name, params in self.parameters.items()}

        # 新增代理模型相关
        self.objectives = objectives
        self.num_objectives = len(objectives)
        self.surrogate = SurrogateModel(num_objectives=self.num_objectives)
        self.observed_perf = {obj: [] for obj in objectives}  # 存储各目标观测值
        self.pareto_front = []  # 存储帕累托解的目标值

    def _encode_config(self, config):
        """将配置编码为数值向量（用于模型输入）"""
        encoded = []
        for name in self.param_names:
            val = config[name]
            param_info = self.parameters[name]
            if self.param_types[name] == 'integer':
                # 归一化到[0,1]
                min_val = param_info['values']['min']
                max_val = param_info['values']['max']
                encoded.append((val - min_val) / (max_val - min_val))
            elif self.param_types[name] == 'float':
                min_val = param_info['values']['min']
                max_val = param_info['values']['max']
                encoded.append((val - min_val) / (max_val - min_val))
            elif self.param_types[name] == 'enum':
                options = param_info['values']
                encoded.append(options.index(val) / (len(options)-1))
            elif self.param_types[name] == 'boolean':
                encoded.append(1.0 if val else 0.0)
        return np.array(encoded)
    
    def update_surrogate(self, window_size: int = 5):
        """更新代理模型"""
        # 准备训练数据
        # recent_data_size = min(len(self.observations),window_size)
        recent_data_size = min(len(self.observations),window_size)
        recent_data = self.observations[-recent_data_size:]

        X = [self._encode_config(c) for c, _ in recent_data]
        y_list = [
            [perf[obj] for _, perf in recent_data] 
            for obj in self.objectives
        ]
        
        # 训练代理模型
        self.surrogate.fit(X, y_list)
        

    def save_surrogate(self, filename):
        self.surrogate.save_model(filename)

    def generate_initial_samples(self, n_samples):
        """使用Sobol序列生成初始配置"""
        dim = len(self.parameters)
        sobol_points = i4_sobol_generate(dim, n_samples)
        configs = []
        for point in sobol_points:
            config = {}
            for i, name in enumerate(self.param_names):
                param_info = self.parameters[name]
                if self.param_types[name] == 'integer':
                    min_val = param_info['values']['min']
                    max_val = param_info['values']['max']
                    config[name] = int(min_val + point[i] * (max_val - min_val))
                elif self.param_types[name] == 'float':
                    min_val = param_info['values']['min']
                    max_val = param_info['values']['max']
                    config[name] = min_val + point[i] * (max_val - min_val)
                elif self.param_types[name] == 'enum':
                    options = param_info['values']
                    idx = int(point[i] * len(options))
                    config[name] = options[min(idx, len(options)-1)]
                elif self.param_types[name] == 'boolean':
                    config[name] = point[i] > 0.5
            config = self.handle_dependency(config)
            configs.append(config)
        return configs

    def generate_configs(self, n_samples=100):
        """生成随机配置样本"""
        configs = []
        for _ in range(n_samples):
            config = {}
            # 遍历字典的键值对 (name, param_info)
            for name, param_info in self.parameters.items():  # 关键修改点
                param_type = param_info['type']  # 从 param_info 获取类型
                if param_type == 'boolean':
                    config[name] = np.random.choice([True, False])
                elif param_type == 'integer':
                    config[name] = np.random.randint(param_info['values']['min'], param_info['values']['max'] + 1)
                elif param_type == 'enum':
                    config[name] = np.random.choice(param_info['values'])
                elif param_type == 'float':
                    config[name] = np.random.uniform(param_info['values']['min'], param_info['values']['max'])
            # config = self.handle_dependency(config)
            configs.append(config)
        return configs
        
    def load_configs_from_history(self, json_path):
        """从json文件读取configs，并补齐缺失参数"""
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # 如果文件里是单个dict，转成list保证一致性
        if isinstance(data, dict):
            data = [data]

        all_configs = []
        for item in data:
            cfg = {}
            raw_config = item.get("config", {})

            for name, param_info in self.parameters.items():
                if name in raw_config:
                    cfg[name] = raw_config[name]
                else:
                    # 随机生成一个合理的值
                    if self.param_types[name] == "integer":
                        min_val = param_info["values"]["min"]
                        max_val = param_info["values"]["max"]
                        cfg[name] = random.randint(min_val, max_val)
                    elif self.param_types[name] == "float":
                        min_val = param_info["values"]["min"]
                        max_val = param_info["values"]["max"]
                        cfg[name] = random.uniform(min_val, max_val)
                    elif self.param_types[name] == "enum":
                        options = param_info["values"]
                        cfg[name] = random.choice(options)
                    elif self.param_types[name] == "boolean":
                        cfg[name] = random.choice([True, False])

            all_configs.append(cfg)

        return all_configs
    
    def handle_dependency(self, config):
        #config['grp-attn-w']=config['grp-attn-n']的整数倍
        if 'grp-attn-n' not in config:
            return config
        max_multiplier = 2048 // config['grp-attn-n'] # 2048是grp-attn-w最大值
        config['grp-attn-w'] = config['grp-attn-n'] * np.random.randint(1, max_multiplier+1)
        return config
    
    def acquisition_ucb(self, config):
        """UCB Acquisition Function"""
        x = self._encode_config(config)
        mu, var = self.surrogate.predict([x])
        sigma = np.sqrt(var[0])
        beta = 2 * np.log((self.iteration**2 * np.pi**2) / (6 * 0.1))
        return mu[0] + beta * sigma

    def suggest_configurations(self, k=1):
        """生成建议配置（核心逻辑）"""
        # 1. 生成候选配置（此处简化：随机生成+约束过滤）
        candidates = self.generate_configs(100)
        
        # 4. 应用Acquisition Function排序（示例使用UCB）
        scores = []
        for config in candidates:
            if self.num_objectives == 1:
                score = self.acquisition_ucb(config)
            else:
                score = self.acquisition_ehvi(config) 
            scores.append(score)
        
        # 选择top-k
        selected_indices = np.argsort(scores)[-k:]
        return [candidates[i] for i in selected_indices]
    
    def update_pareto_front(self):
        """更新Pareto前沿（每次新观测后调用）"""
        new_front = []
        # 筛选可行解
        feasible_solutions = [
            (config, perf) 
            for config, perf in self.observations
        ]
        
        # 支配关系计算
        for candidate in feasible_solutions:
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

    def set_reference_point(self):
        ref_point = []
        for obj, direction in self.objectives.items():
            # 提取当前目标的所有观测值（通过目标名称 obj）
            values = [perf[obj] for _, perf in self.observations]
            
            # 根据优化方向确定参考点
            if direction == "min":
                # 最小化目标：参考点取最大值（确保被所有解支配）
                ref_val = max(values)
            else:
                # 最大化目标：参考点取最小值（确保被所有解支配）
                ref_val = min(values)
            ref_point.append(ref_val)
        
        self.reference_point = ref_point

    def get_pareto_front(self) -> List[Tuple[Dict, Dict]]:
        """获取当前Pareto前沿（配置+指标）"""
        return self.pareto_front

    def acquisition_ehvi(self, config):
        """EHVI Acquisition Function (Expected Hypervolume Improvement)"""
        x = self._encode_config(config)
        preds = self.surrogate.predict([x])
        means = np.array([mu[0] for mu, _ in preds])
        stds = np.array([sigma[0] for _, sigma in preds])

        # 如果没有有效参考点或Pareto前沿，返回0
        if not hasattr(self, 'reference_point') or not self.pareto_front:
            return 0.0

        # 当前Pareto前沿目标值（转为数组）
        front_y = np.array([
            [perf[obj] for obj in self.objectives] 
            for _, perf in self.pareto_front
        ])

        # pymoo 的 HV 指标期望的是 minimization 问题，
        # 所以我们需要统一方向（最大化指标取负）
        obj_dirs = [1 if self.objectives[obj] == 'min' else -1 for obj in self.objectives]
        front_y = front_y * obj_dirs
        ref_point = np.array(self.reference_point) * obj_dirs
        mu_scaled = means * obj_dirs
        std_scaled = stds

        # Monte Carlo 估计 EHVI（简化版）
        samples = np.random.normal(loc=mu_scaled, scale=std_scaled, size=(128, self.num_objectives))
        hv = HV(ref_point=ref_point)

        hvs = []
        for s in samples:
            extended = np.vstack([front_y, s])
            hvs.append(hv.do(extended))

        current_hv = hv.do(front_y)
        ehvi_estimate = np.mean([h - current_hv for h in hvs])
        return ehvi_estimate

    def evaluate_pareto(self, perfs):
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



if __name__ == "__main__":
    # 示例参数配置
    parameters_path = "path/to/parameters.json"  # 替换为实际路径
    objectives = {'tps_avg': 'max', 'gpu_avg': 'min'}  # 示例目标
    tuner = LaTune(parameters_path, objectives)

    # 生成初始样本
    initial_samples = tuner.generate_initial_samples(10)
    print("Initial Samples:", initial_samples)

    # 更新代理模型
    tuner.update_surrogate()

    # 生成建议配置
    suggested_configs = tuner.suggest_configurations(k=5)
    print("Suggested Configurations:", suggested_configs)
