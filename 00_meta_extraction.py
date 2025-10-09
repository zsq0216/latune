import re
import numpy as np
from llama_executor import LlamaExecutor
import time
import os
import json
import argparse


class MetaFeatureExtractor:
    """
    Extracts, normalizes, and vectorizes meta features from LLM runtime logs,
    and compares them against historical records using an Epanechnikov kernel
    similarity measure.

    This class is designed to:
      1) Parse raw log text to collect model, quantization, and hardware stats.
      2) Normalize those features into comparable scales.
      3) Build a feature vector grouped by category (model / hardware).
      4) Save and load JSONL records for later nearest-neighbor style lookups.

    Notes
    -----
    - The normalization constants are heuristic and may need tuning for your
      specific models and hardware.
    - JSONL persistence is append-only; each line is an independent record.
    - Similarity uses the Epanechnikov quadratic kernel over Lp distances.
    """

    def __init__(self, model_name, hardware, filepath="meta_features/records.jsonl"):
        # Group weights let you scale contribution by category if needed.
        self.group_weights = {
            "model": 1,
            "hardware": 1,
        }
        self.features = {}
        self.norm = {}
        self.vector = []
        self.model_name = model_name
        self.hardware = hardware
        self.filepath = filepath

    @staticmethod
    def _now_ts():
        """Return the current UNIX timestamp (seconds)."""
        return int(time.time())

    @staticmethod
    def _to_list(x):
        """Convert numpy arrays to plain Python lists for JSON serialization."""
        if isinstance(x, np.ndarray):
            return x.astype(float).tolist()
        return list(x)

    @staticmethod
    def _from_list(x):
        """Convert a list back to a numpy float32 array."""
        return np.array(x, dtype=np.float32)

    def parse_log(self, log_text: str):
        """Extract key fields from a raw log text block."""
        features = {}

        # --- Model architecture ---
        features["n_params_billion"] = self._extract_float(r"model params\s*=\s*([\d\.]+)", log_text)
        features["n_layer"] = self._extract_int(r"n_layer\s*=\s*(\d+)", log_text)
        features["n_embd"] = self._extract_int(r"n_embd\s*=\s*(\d+)", log_text)
        features["n_head"] = self._extract_int(r"n_head\s*=\s*(\d+)", log_text)
        features["n_head_kv"] = self._extract_int(r"n_head_kv\s*=\s*(\d+)", log_text)
        features["n_ff"] = self._extract_int(r"n_ff\s*=\s*(\d+)", log_text)
        features["context_len_train"] = self._extract_int(r"n_ctx_train\s*=\s*(\d+)", log_text)

        # Treat model as MoE if the number of experts is greater than zero.
        features["is_moe"] = 1 if self._extract_int(r"print_info:\s*n_expert\s*=\s*(\d+)", log_text) > 0 else 0

        # Quantization / file info
        features["quant_file_type"] = self._extract_str(r"file type\s*=\s*(\S+)", log_text)
        m = re.search(r"file size\s*=\s*([\d\.]+)\s*GiB\s*\(([\d\.]+)\s*BPW\)", log_text)
        # NOTE: This assumes the pattern exists; consider adding guards if logs may omit it.
        features["file_size_GiB"] = float(m.group(1))
        features["bpw"] = float(m.group(2))

        # --- Hardware ---
        # GPU free memory is parsed from MiB; converted to GB by dividing by 1024.
        features["gpu_mem_GB"] = self._extract_float(r"(\d+) MiB free", log_text) / 1024
        features["cpu_threads"] = self._extract_int(r"n_threads\s*=\s*(\d+)", log_text)
        # Use provided hardware label as the GPU name (maps to a vendor score later).
        features["gpu_name"] = self.hardware

        self.features = features
        return features

    def normalize(self, features: dict):
        """
        Normalize numeric features into roughly [0, 1] ranges using simple,
        heuristic upper bounds. Adjust these scales to match your environment.
        """
        norm = {}
        # Model architecture (heuristic denominators)
        norm["n_params_billion"] = features["n_params_billion"] / 10       # assume 10B upper bound
        norm["n_layer"] = features["n_layer"] / 100
        norm["n_embd"] = features["n_embd"] / 8192
        norm["n_head"] = features["n_head"] / 128
        norm["n_head_kv"] = features["n_head_kv"] / 64
        norm["n_ff"] = features["n_ff"] / 65536
        norm["context_len_train"] = features["context_len_train"] / 65536
        norm["is_moe"] = features["is_moe"] / 5  # small scale to dampen impact

        # Quantization type: coarse mapping; extend as needed.
        qtype = features.get("quant_file_type", "").upper()
        if "Q4" in qtype:
            norm["quant_file_type"] = 0.4
        elif "Q8" in qtype:
            norm["quant_file_type"] = 0.6
        else:
            norm["quant_file_type"] = 0.5

        norm["file_size_GiB"] = features["file_size_GiB"] / 32.0
        norm["bpw"] = features["bpw"] / 32.0

        # Hardware (heuristic denominators)
        norm["gpu_mem_GB"] = features["gpu_mem_GB"] / 32     # assume 32 GB as an upper bound
        norm["cpu_threads"] = features["cpu_threads"] / 32    # assume 32 threads

        # Map common device labels to a vendor score. Extend this as needed.
        gpu_map = {"orin": 0.2, "m4": 0.3, "rtx3060": 0.4, "rtx4090": 0.6}
        norm["gpu_vendor"] = gpu_map.get(features["gpu_name"], 0)

        self.norm = norm
        return norm

    def to_vector(self, norm_features: dict):
        """
        Concatenate normalized features into a single vector, grouped by category
        and weighted by self.group_weights.
        """
        # Model-related keys (order defines the vector layout)
        model_keys = [
            "n_params_billion",
            "n_layer",
            "n_embd",
            "n_head",
            "n_head_kv",
            "n_ff",
            "context_len_train",
            # MoE metadata
            "is_moe",
            # Quantization / file metadata
            "quant_file_type",
            "file_size_GiB",
            "bpw",
        ]

        # Hardware-related keys
        hardware_keys = [
            "gpu_mem_GB",
            "cpu_threads",
            "gpu_vendor"
        ]

        # Reserved for future workload-specific features (not used yet)
        workload_keys = [
            "task_type",     # enum-like: {"chat", "doc_summary"} (placeholder)
            "avg_input_len"
        ]

        # Concatenate and apply group weights
        vector = []
        for k in model_keys:
            vector.append(norm_features[k] * self.group_weights["model"])
        for k in hardware_keys:
            vector.append(norm_features[k] * self.group_weights["hardware"])

        self.vector = np.array(vector, dtype=np.float32)
        return vector

    def _extract_int(self, pattern, text):
        """Return the first captured integer for pattern, or 0 if not found."""
        m = re.search(pattern, text)
        return int(m.group(1)) if m else 0

    def _extract_float(self, pattern, text):
        """Return the first captured float for pattern, or 0.0 if not found."""
        m = re.search(pattern, text)
        return float(m.group(1)) if m else 0.0

    def _extract_str(self, pattern, text, default=""):
        """Return the first captured string for pattern, or default if not found."""
        m = re.search(pattern, text)
        return m.group(1).strip() if m else default

    def _epanechnikov_similarity(self, x, y, h=1.0, p=2):
        """
        Epanechnikov quadratic kernel similarity.

        Parameters
        ----------
        x, y : np.ndarray
            Feature vectors of the same shape.
        h : float
            Bandwidth parameter. Larger values smooth distances more.
        p : int
            Norm order for the Lp distance (default L2).

        Returns
        -------
        float
            Similarity in [0, 0.75], where values close to 0.75 indicate high similarity.
            Returns 0.0 if the scaled distance exceeds 1.
        """
        dist = np.linalg.norm(x - y, ord=p) / h
        if dist <= 1:
            return 0.75 * (1 - dist**2)
        else:
            return 0.0

    def characterize_feature(self, result):
        """
        High-level convenience method:
          - parse -> normalize -> vectorize
        Returns the final feature vector.
        """
        self.parse_log(result)
        self.normalize(self.features)
        self.to_vector(self.norm)
        return self.vector

    def save_record(self):
        """
        Append the current record to a JSONL file (one JSON object per line).

        The record includes:
          - model_name
          - hardware
          - raw extracted features
          - normalized features
          - vector (as a list)
        """
        os.makedirs(os.path.dirname(os.path.abspath(self.filepath)), exist_ok=True)
        record = {
            "model_name": self.model_name,
            "hardware": self.hardware,
            "features": self.features,            # raw extracted fields
            "norm": self.norm,                    # normalized features
            "vector": self._to_list(self.vector)  # numpy array -> list for JSON
        }
        with open(self.filepath, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
        return True

    def load_and_compare(self, h=1.0, p=2, top_k=None):
        """
        Load historical records from the JSONL file and compute Epanechnikov
        similarities against the current vector.

        Returns a list sorted by similarity (descending), each item containing:
          {
            "model_name": str,
            "hardware": str,
            "similarity": float,
            "vector_len": int
          }

        Parameters
        ----------
        h : float
            Bandwidth for similarity kernel.
        p : int
            Lp norm order for distance.
        top_k : int or None
            If provided, return only the top-k matches.
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
                    # Skip records whose vector shape does not match current features.
                    if vec.size == 0 or vec.shape != curr.shape:
                        # Consider implementing alignment/truncation/padding if needed.
                        continue
                    sim = self._epanechnikov_similarity(curr, vec, h=h, p=p)
                    results.append({
                        "model_name": rec.get("model_name", ""),
                        "hardware": rec.get("hardware", ""),
                        "similarity": float(sim),
                        "vector_len": int(vec.size),
                    })
                except json.JSONDecodeError:
                    # Ignore malformed lines.
                    continue

        results.sort(key=lambda x: x["similarity"], reverse=True)
        if top_k is not None:
            results = results[:top_k]
        return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Llama Configuration Optimizer")
    parser.add_argument(
        "--device",
        type=str,
        choices=["cpu", "gpu"],
        default="gpu",
        help="Processing device (cpu or gpu).",
    )
    parser.add_argument(
        "--hardware",
        type=str,
        choices=["rtx3060", "rtx4090", "m4", "orin"],
        default="rtx3060",
        help="Processing hardware label.",
    )
    parser.add_argument(
        "--model",
        type=str,
        choices=["qwen3-4b", "phimoe-mini"],
        default="qwen3-4b",
        help="Model family (e.g., qwen3-8b, phimoe-mini).",
    )
    parser.add_argument(
        "--quant",
        type=str,
        choices=["q4", "q8"],
        default="q4",
        help="Quantization tag (q4, q8).",
    )
    args = parser.parse_args()

    # Example executor configuration for demonstration purposes.
    param_types_instance = {"gpu-layers": "integer"}
    config = {"gpu-layers": 32}
    model_name = f"{args.model}-{args.quant}"

    extractor = MetaFeatureExtractor(model_name=model_name, hardware=args.hardware)
    executor = LlamaExecutor(param_types=param_types_instance, device="gpu")

    print(config)
    # Expect executor.extract_meta_feature to return a log text blob compatible with parse_log.
    result = executor.extract_meta_feature(config, model_path=f"./../models/{model_name}.gguf")
    vector = extractor.characterize_feature(result)

    results = extractor.load_and_compare(top_k=3)
    print(results)
    if extractor.save_record():
        print("Successfully saved!")
