import json
import numpy as np
import pandas as pd
import shap
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import re
import matplotlib.pyplot as plt
from llama_executor import LlamaExecutor
import argparse
import os
from collections import defaultdict

class LlamaConfigOptimizer:
    def __init__(self, params_path, device="gpu",model="./../models/qwen3-4b-q4.gguf", model_name="qwen3-4b-q4", hardware="m4"):
        self.params_path = params_path
        self.device = device
        self.model = model
        self.model_name = model_name
        self.hardware = hardware
        self.X = None
        self.y = None
        self.performance_params = self._load_params()
        self.param_types = {name: param["type"] for name, param in self.performance_params.items()}
        self.executor = LlamaExecutor(self.param_types, model_path=model, device=device,hardware=hardware)

        # 新增：在预处理时会填充
        self.param_feature_map = None   # {base_param: [feature_cols]}
        self.processed_X_cache = None

    def _load_params(self):
        with open(self.params_path, "r") as f:
            return json.load(f)

    def evaluate_config(self, config):
        results = self.executor.run_server_performance_test(config)
        return {
            "tps_avg": results.get("tps_avg", 0.0),
            "gpu_avg": results.get("gpu_avg", 0.0)
        }

    def generate_dataset(self, n_samples=50):
        configs = pd.DataFrame(self.executor.generate_configs(self.performance_params, n_samples))
        self.X = configs
        results_list = []

        for i, config in configs.iterrows():
            metrics = self.evaluate_config(config.to_dict())
            results_list.append(metrics)
            print(f"i: {i}, metrics: {metrics}")

        self.y = pd.DataFrame(results_list)
        return self.X, self.y

    def _preprocess_data(self):
        """
        - 布尔转 0/1
        - 枚举做 one-hot
        - 同时构建 param_feature_map：把独热特征映射回原始参数名，后续用于按参数聚合打分
        """
        if self.processed_X_cache is not None and self.param_feature_map is not None:
            return self.processed_X_cache

        X = self.X.copy()
        param_feature_map = defaultdict(list)

        for name, param_info in self.performance_params.items():
            if name not in X.columns:
                continue
            if param_info["type"] == "boolean":
                X[name] = X[name].astype(int)
                param_feature_map[name].append(name)
            elif param_info["type"] == "enum":
                # 记录当前已有列，便于拿到新生成的独热列名
                before_cols = set(X.columns)
                X = pd.get_dummies(X, columns=[name], prefix=name, prefix_sep="__")
                new_cols = [c for c in X.columns if c not in before_cols]
                # 独热特征全部归属该枚举参数
                for c in new_cols:
                    param_feature_map[name].append(c)
            else:
                # integer / float 等连续型
                param_feature_map[name].append(name)

        # 有些参数可能没出现在生成的 configs（极端情况），也填入映射（空列表）
        for name in self.performance_params.keys():
            param_feature_map.setdefault(name, [name] if name in X.columns else [])

        self.param_feature_map = dict(param_feature_map)
        self.processed_X_cache = X
        return X

    def train_model(self, test_size=0.2):
        processed_X = self._preprocess_data()
        self.models = {}

        for target in self.y.columns:
            X_train, X_test, y_train, y_test = train_test_split(
                processed_X, self.y[target], test_size=test_size, random_state=42
            )
            model = RandomForestRegressor()
            model.fit(X_train, y_train)
            self.models[target] = {
                'model': model,
                'X_test': X_test,
                'y_test': y_test
            }
        return self.models

    def analyze_shap(self, output_dir="shap_outputs"):
        os.makedirs(output_dir, exist_ok=True)
        processed_X = self._preprocess_data()
        shap_values_dict = {}

        for target, model_info in self.models.items():
            model = model_info["model"]
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(processed_X)
            shap_values_dict[target] = shap_values

            # 保存图像
            plt.figure()
            shap.summary_plot(shap_values, processed_X, plot_type="bar", show=False)
            plt.savefig(f"{output_dir}/{self.hardware}/bar_{target}_{self.model_name}.png", bbox_inches='tight', dpi=300)
            plt.close()

            plt.figure()
            shap.summary_plot(shap_values, processed_X, show=False)
            plt.savefig(f"{output_dir}/{self.hardware}/{target}_{self.model_name}.png", bbox_inches='tight', dpi=300)
            plt.close()

        return shap_values_dict

    def calculate_tunability(self, shap_values_dict):
        """
        按特征列（经过预处理后的列）计算“正 SHAP 的平均值”作为可调性，并给出每个 target 的 Series
        """
        processed_X = self._preprocess_data()
        tunability_scores = {}

        for target, shap_values in shap_values_dict.items():
            shap_df = pd.DataFrame(shap_values, columns=processed_X.columns)
            tunability = shap_df.apply(lambda x: x[x > 0].mean()).sort_values(ascending=False)
            tunability_scores[target] = tunability

        return tunability_scores

    # ===== 新增：加权得分（参数级别） =====
    def compute_weighted_param_scores(self, tunability_scores, weights=None):
        """
        将不同目标(metric)的可调性按权重加权：
            score(param) = 0.7 * tunability_tps(param) + 0.3 * tunability_gpu_mem(param)
        其中对枚举产生的独热特征，先在特征层面加权，再聚合回原始参数。
        """
        if weights is None:
            weights = {"tps_avg": 0.7, "gpu_avg": 0.3}

        processed_X = self._preprocess_data()
        feature_index = processed_X.columns

        # 先把各指标的特征级可调性对齐到相同索引
        combined_feature_scores = pd.Series(0.0, index=feature_index)
        available_metrics = []

        for metric, series in tunability_scores.items():
            if metric not in weights:
                continue
            available_metrics.append(metric)
            w = float(weights[metric])
            aligned = series.reindex(feature_index).fillna(0.0)
            combined_feature_scores = combined_feature_scores.add(aligned * w, fill_value=0.0)

        if not available_metrics:
            raise ValueError("No available metrics found for weighting. "
                             f"Expected any of {list(weights.keys())}, got {list(tunability_scores.keys())}.")

        # 聚合成“参数级”分数（把所有属于该参数的独热/特征列求和）
        param_scores = {}
        for base_param, feat_cols in self.param_feature_map.items():
            if not feat_cols:
                param_scores[base_param] = 0.0
            else:
                param_scores[base_param] = float(combined_feature_scores.reindex(feat_cols).fillna(0.0).sum())

        param_scores = pd.Series(param_scores).sort_values(ascending=False)
        return param_scores

    # ===== 新增：写回带 rank 的 JSON =====
    def write_ranked_params_json(self, ranked_params: pd.Series, output_path: str):
        """
        在原 knobs JSON 的基础上加上 "rank" 字段，并按排名高低排序后输出。
        - ranked_params 是一个按得分降序的 Series，index 为参数名
        """
        original = self._load_params()

        # 生成 rank 映射：参数 -> 排名（从 1 开始）
        rank_map = {}
        rank_counter = 1
        for p in ranked_params.index:
            rank_map[p] = rank_counter
            rank_counter += 1

        # 没出现在 ranked_params 的参数（极少情况），排在最后
        for p in original.keys():
            if p not in rank_map:
                rank_map[p] = rank_counter
                rank_counter += 1

        # 重新组织输出：按 rank 排序
        sorted_items = sorted(original.items(), key=lambda kv: rank_map[kv[0]])

        out_dict = {}
        for p, meta in sorted_items:
            meta_copy = dict(meta)  # 避免修改原对象
            meta_copy["rank"] = int(rank_map[p])
            out_dict[p] = meta_copy

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(out_dict, f, ensure_ascii=False, indent=4)

        return output_path

    def save_metric_extrema_from_memory(self, model: str, output_path: str = None):
        """
        直接使用 self.y 计算各指标的 min/max 并保存为 JSON 文件。
        """
        if self.y is None:
            raise ValueError("还没有生成数据集，请先调用 generate_dataset。")

        extrema = {}
        metric_candidates = ["tps_avg", "gpu_avg"]

        for m in metric_candidates:
            if m in self.y.columns:
                s = pd.to_numeric(self.y[m], errors="coerce")
                extrema[m] = {
                    "min": float(np.nanmin(s.values)) if len(s) else None,
                    "max": float(np.nanmax(s.values)) if len(s) else None
                }

        if not extrema:
            raise ValueError(f"self.y 中没有找到指标列（候选: {metric_candidates}）。")

        if output_path is None:
            output_path = f"bounds/{self.hardware}/{model}.json"

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(extrema, f, ensure_ascii=False, indent=4)

        return output_path

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

    model_name = f"{args.model}-{args.quant}"  # 模型名称

    optimizer = LlamaConfigOptimizer(
        params_path=f"knobs_files/knobs_raw.json",
        device=args.device,
        model = f"./../models/{model_name}.gguf",
        model_name = model_name,
        hardware=args.hardware
    )

    # 生成数据集
    X, y = optimizer.generate_dataset(n_samples=50)
    os.makedirs("shap_outputs", exist_ok=True)

    extrema_json_path = optimizer.save_metric_extrema_from_memory(model_name)
    print(f"Saved metric extrema JSON to: {extrema_json_path}")

    X.to_csv(f"shap_outputs/{args.hardware}/X_{model_name}.csv", index=False)
    y.to_csv(f"shap_outputs/{args.hardware}/y_{model_name}.csv", index=False)

    # 训练模型
    optimizer.train_model()

    # 分析 SHAP
    shap_values_dict = optimizer.analyze_shap()

    # 计算 tunability（按每个指标得到特征级可调性）
    tunability_scores = optimizer.calculate_tunability(shap_values_dict)

    # ===== 新增：计算加权得分（tps*0.7 + gpu_mem*0.3），聚合为“参数级”并排序 =====
    weighted_param_scores = optimizer.compute_weighted_param_scores(
        tunability_scores,
        weights={"tps_avg": 0.7, "gpu_avg": 0.3}
    )

    print("\nWeighted Param Scores (desc):")
    print(weighted_param_scores)

    # ===== 新增：写 JSON（在原有基础上加 'rank'）=====
    output_json = f"knobs_files/{args.hardware}/{model_name}.json"
    path = optimizer.write_ranked_params_json(weighted_param_scores, output_json)
    print(f"\nWrote ranked params JSON to: {path}")

