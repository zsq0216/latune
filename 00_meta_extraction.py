import re
import numpy as np
from llama_executor import LlamaExecutor
import time
import os
import json
import argparse

class MetaFeatureExtractor:
    def __init__(self, model_name, hardware, filepath = "meta_features/records.jsonl"):
        # 字段分组及其手工权重
        self.group_weights = {
            "model": 1,
            # "runtime": 0.4,
            "hardware": 1,
            # "decoding": 0.1,
        }
        self.features = {}
        self.norm = {}
        self.vector = []
        self.model_name = model_name
        self.hardware = hardware
        self.filepath = filepath

    @staticmethod
    def _now_ts():
        return int(time.time())

    @staticmethod
    def _to_list(x):
        if isinstance(x, np.ndarray):
            return x.astype(float).tolist()
        return list(x)

    @staticmethod
    def _from_list(x):
        return np.array(x, dtype=np.float32)
    
    def parse_log(self, log_text: str):
        """从日志文本中提取关键字段"""
        features = {}
        # print(log_text)

        # --- 模型结构 ---
        features["n_params_billion"] = self._extract_float(r"model params\s*=\s*([\d\.]+)", log_text)
        features["n_layer"] = self._extract_int(r"n_layer\s*=\s*(\d+)", log_text)
        features["n_embd"] = self._extract_int(r"n_embd\s*=\s*(\d+)", log_text)
        features["n_head"] = self._extract_int(r"n_head\s*=\s*(\d+)", log_text)
        features["n_head_kv"] = self._extract_int(r"n_head_kv\s*=\s*(\d+)", log_text)
        features["n_ff"] = self._extract_int(r"n_ff\s*=\s*(\d+)", log_text)
        features["context_len_train"] = self._extract_int(r"n_ctx_train\s*=\s*(\d+)", log_text)
        features["is_moe"] = 1 if self._extract_int(r"print_info:\s*n_expert\s*=\s*(\d+)", log_text) > 0 else 0
        features["quant_file_type"] = self._extract_str(r"file type\s*=\s*(\S+)", log_text)
        m = re.search(r"file size\s*=\s*([\d\.]+)\s*GiB\s*\(([\d\.]+)\s*BPW\)", log_text)
        features["file_size_GiB"] = float(m.group(1))
        features["bpw"] = float(m.group(2))

        # --- 硬件 ---
        features["gpu_mem_GB"] = self._extract_float(r"(\d+) MiB free", log_text) / 1024
        features["cpu_threads"] = self._extract_int(r"n_threads\s*=\s*(\d+)", log_text)
        features["gpu_name"] = self._extract_str(r"GPU name:\s*(.+)", log_text)

        self.features = features

    def normalize(self, features: dict):
        """归一化数值 (简单缩放，按经验设上限)"""
        norm = {}
        norm["n_params_billion"] = features["n_params_billion"] / 10      # 假设上限 10B
        norm["n_layer"] = features["n_layer"] / 100
        norm["n_embd"] = features["n_embd"] / 8192
        norm["n_head"] = features["n_head"] / 128
        norm["n_head_kv"] = features["n_head_kv"] / 64
        norm["n_ff"] = features["n_ff"] / 65536
        norm["context_len_train"] = features["context_len_train"] / 65536
        norm["is_moe"] = features["is_moe"]/5

        qtype = features.get("quant_file_type", "").upper()  # 转大写方便匹配
        if "Q4" in qtype:
            norm["quant_file_type"] = 0.4
        elif "Q8" in qtype:
            norm["quant_file_type"] = 0.6
        else:
            norm["quant_file_type"] = 0.5

        norm["file_size_GiB"] = features["file_size_GiB"] / 32.0
        norm["bpw"] = features["bpw"] / 32.0

        norm["gpu_mem_GB"] = features["gpu_mem_GB"] / 32       # 假设 32GB 上限
        norm["cpu_threads"] = features["cpu_threads"] / 32    # 假设 32 核

        name = (features.get("gpu_name") or "").lower()

        # 厂商 one-hot
        is_apple  = 1.0 if "apple"  in name else 0.0
        is_nvidia = 1.0 if any(k in name for k in ["nvidia", "geforce", "rtx", "tesla", "quadro"]) else 0.0
        is_amd    = 1.0 if any(k in name for k in ["amd", "radeon", "instinct"]) else 0.0

        base_affinity = 0.6  # unknown
        if is_nvidia:
            base_affinity = 0.9
        elif is_amd:
            base_affinity = 0.75
        elif is_apple:
            base_affinity = 0.7

        norm["gpu_vendor"] = base_affinity
        self.norm = norm

    def to_vector(self, norm_features: dict):
        """按分组 + 权重拼接成向量"""
        # 分组
        model_keys = [
            "n_params_billion",
            "n_layer",
            "n_embd",
            "n_head",
            "n_head_kv",
            "n_ff",
            "context_len_train",
            # MoE 元信息
            "is_moe",
            # 量化/文件元信息
            "quant_file_type",
            "file_size_GiB",
            "bpw",
        ]
        
        hardware_keys = [
            "gpu_mem_GB",
            "cpu_threads",
            "gpu_vendor"
        ]

        # 拼接并加权
        vector = []
        for k in model_keys:
            vector.append(norm_features[k] * self.group_weights["model"])
        for k in hardware_keys:
            vector.append(norm_features[k] * self.group_weights["hardware"])

        self.vector = np.array(vector, dtype=np.float32)

    def _extract_int(self, pattern, text):
        m = re.search(pattern, text)
        return int(m.group(1)) if m else 0

    def _extract_float(self, pattern, text):
        m = re.search(pattern, text)
        return float(m.group(1)) if m else 0.0
    
    def _extract_str(self, pattern, text, default=""):
        m = re.search(pattern, text)
        return m.group(1).strip() if m else default
    
    def _epanechnikov_similarity(self, x, y, h=1.0, p=2):
        """
        Epanechnikov quadratic kernel similarity
        x, y: 向量
        h: 带宽
        p: 距离的范数 (默认 L2)
        """
        dist = np.linalg.norm(x - y, ord=p) / h
        if dist <= 1:
            return 0.75 * (1 - dist**2)
        else:
            return 0.0
        
    def characterize_feature(self, result):
        self.parse_log(result)
        self.normalize(self.features)
        self.to_vector(self.norm)
        return self.vector

    
    def save_record(self):
        """
        以 JSONL 方式追加保存一条记录（每行一个 JSON）。
        - filepath: 建议以 .jsonl 结尾
        """
        os.makedirs(os.path.dirname(os.path.abspath(self.filepath)), exist_ok=True)
        record = {
            "timestamp": self._now_ts(),
            "model_name": self.model_name,
            "hardware": self.hardware,
            "features": self.features,           # 原始提取
            "norm": self.norm,                   # 归一化特征
            "vector": self._to_list(self.vector) # 向量转 list 便于存盘
        }
        with open(self.filepath, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
        return True

    def load_and_compare(self, h=1.0, p=2, top_k=None):
        """
        从 JSONL 文件读取历史记录，与 current_vector 逐条计算 Epanechnikov 相似度。
        返回按相似度从高到低排序的列表，每项包含：
        {model_name, hardware, timestamp, similarity, vector_len}
        - top_k: 只返回前 k 个（None 表示全部）
        """
        if not os.path.exists(self.filepath):
            raise FileNotFoundError(f"No such file: {self.filepath}")

        curr = np.asarray(self.vector, dtype=np.float32)
        results = []
        with open(self.filepath, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                    vec = self._from_list(rec.get("vector", []))
                    if vec.size == 0 or vec.shape != curr.shape:
                        # 维度不匹配时跳过（也可在此做对齐/截断/补零策略）
                        continue
                    sim = self._epanechnikov_similarity(curr, vec, h=h, p=p)
                    results.append({
                        "model_name": rec.get("model_name", ""),
                        "hardware": rec.get("hardware", ""),
                        "timestamp": rec.get("timestamp", 0),
                        "similarity": float(sim),
                        "vector_len": int(vec.size),
                    })
                except json.JSONDecodeError:
                    # 跳过坏行
                    continue

        results.sort(key=lambda x: x["similarity"], reverse=True)
        if top_k is not None:
            results = results[:top_k]
        return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Llama Configuration Optimizer')
    parser.add_argument('--device', type=str, choices=['cpu', 'gpu'], default='gpu',
                        help='Processing device (cpu or gpu)')
    parser.add_argument('--hardware', type=str, choices=['rtx3060', 'rtx4090', 'm4', 'orin'], default='m4',
                       help='Processing hardware')
    parser.add_argument('--model', type=str, choices=['qwen3-4b','phimoe-mini'], default='phimoe-mini',
                        help='qwen3-8b, phimoe-mini')
    parser.add_argument('--quant', type=str, choices=['q4','q8'],default='q4',
                        help='q4, q8')
    args = parser.parse_args()
    param_types_instance ={'gpu-layers': 'integer'}
    config ={"gpu-layers": 32}
    model_name = f"{args.model}-{args.quant}"

    extractor = MetaFeatureExtractor(model_name=model_name, hardware = args.hardware)
    executor = LlamaExecutor(param_types=param_types_instance,
                              device="gpu")

    # print(config)
    result = executor.extract_meta_feature(config, model_path=f"./../models/{args.hardware}_{model_name}.gguf")
    vector = extractor.characterize_feature(result)

    results = extractor.load_and_compare(top_k=3)
    print(results)
    if extractor.save_record():
        print("Successfully saved!")
