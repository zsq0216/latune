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
    """
    Trains simple performance models for LLM serving configs, analyzes feature
    importance with SHAP, and ranks tunable parameters.

    Workflow:
      1) Generate candidate configurations via `LlamaExecutor`.
      2) Evaluate each config to collect performance metrics (e.g., TPS, GPU usage).
      3) Preprocess features (booleans, enums, continuous) and train a model per metric.
      4) Compute SHAP values and derive "tunability" scores.
      5) Aggregate feature-level scores back to parameter level, weight across metrics,
         and emit a ranked parameter JSON.

    Notes
    -----
    - Parameters schema is loaded from a JSON file (knobs definition).
    - Enum parameters are one-hot encoded; mapping back to base parameters is tracked.
    - Plots and outputs are saved to disk; ensure directories exist or are creatable.
    """

    def __init__(
        self,
        params_path,
        device="gpu",
        model="./../models/qwen3-4b-q4.gguf",
        model_name="qwen3-4b-q4",
        hardware="m4",
    ):
        self.params_path = params_path
        self.device = device
        self.model = model
        self.model_name = model_name
        self.hardware = hardware

        self.X = None
        self.y = None

        # Load knobs definitions and derive param types for the executor.
        self.performance_params = self._load_params()
        self.param_types = {
            name: param["type"] for name, param in self.performance_params.items()
        }

        # Executor used for config generation and performance evaluation.
        self.executor = LlamaExecutor(
            self.param_types, model_path=model, device=device, hardware=hardware
        )

        # Filled during preprocessing:
        # - param_feature_map: maps base parameter -> list of processed feature columns
        # - processed_X_cache: cached preprocessed feature matrix
        self.param_feature_map = None  # {base_param: [feature_cols]}
        self.processed_X_cache = None

    def _load_params(self):
        """Load knobs schema (JSON) describing parameters and their types."""
        with open(self.params_path, "r") as f:
            return json.load(f)

    def evaluate_config(self, config):
        """
        Run a server performance test for a single configuration via the executor.

        Returns
        -------
        dict
            Contains metrics such as:
              - "tps_avg": average tokens-per-second (float)
              - "gpu_avg": average GPU utilization or memory metric (float)
        """
        results = self.executor.run_server_performance_test(config)
        return {
            "tps_avg": results.get("tps_avg", 0.0),
            "gpu_avg": results.get("gpu_avg", 0.0),
        }

    def generate_dataset(self, n_samples=50):
        """
        Generate a dataset of random configurations and their measured performance.

        Parameters
        ----------
        n_samples : int
            Number of configurations to sample and evaluate.

        Returns
        -------
        (pd.DataFrame, pd.DataFrame)
            X: configurations (raw parameter values)
            y: metrics (e.g., tps_avg, gpu_avg)
        """
        configs = pd.DataFrame(
            self.executor.generate_configs(self.performance_params, n_samples)
        )
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
        Preprocess features with the following rules:
          - Booleans -> 0/1
          - Enums -> one-hot encoding
          - Continuous (int/float) -> unchanged

        Also builds `param_feature_map` to trace processed columns back to their
        original base parameter names. This enables aggregation from feature-level
        SHAP/tunability scores back to parameter-level scores.

        Returns
        -------
        pd.DataFrame
            Processed feature matrix suitable for training.
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
                # Record current columns so we can capture newly created one-hot columns.
                before_cols = set(X.columns)
                X = pd.get_dummies(X, columns=[name], prefix=name, prefix_sep="__")
                new_cols = [c for c in X.columns if c not in before_cols]
                # All one-hot columns belong to the same base enum parameter.
                for c in new_cols:
                    param_feature_map[name].append(c)

            else:
                # integer / float (continuous)
                param_feature_map[name].append(name)

        # Ensure every parameter appears in the mapping (even if absent in X).
        for name in self.performance_params.keys():
            param_feature_map.setdefault(name, [name] if name in X.columns else [])

        self.param_feature_map = dict(param_feature_map)
        self.processed_X_cache = X
        return X

    def train_model(self, test_size=0.2):
        """
        Train a RandomForestRegressor per target metric in `self.y`.

        Parameters
        ----------
        test_size : float
            Fraction of data used as test split (for basic sanity checks).

        Returns
        -------
        dict
            Mapping from target metric -> dict with keys:
              - "model": trained RandomForestRegressor
              - "X_test": test features
              - "y_test": test targets
        """
        processed_X = self._preprocess_data()
        self.models = {}

        for target in self.y.columns:
            X_train, X_test, y_train, y_test = train_test_split(
                processed_X, self.y[target], test_size=test_size, random_state=42
            )
            model = RandomForestRegressor()
            model.fit(X_train, y_train)
            self.models[target] = {"model": model, "X_test": X_test, "y_test": y_test}

        return self.models

    def analyze_shap(self, output_dir="shap_outputs"):
        """
        Compute SHAP values for each trained model and save summary plots.

        Outputs
        -------
        - Bar summary plot for global importance per metric.
        - Beeswarm summary plot for feature distributions.
        Files are saved under: {output_dir}/{hardware}/
        """
        os.makedirs(output_dir, exist_ok=True)
        processed_X = self._preprocess_data()
        shap_values_dict = {}

        for target, model_info in self.models.items():
            model = model_info["model"]
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(processed_X)
            shap_values_dict[target] = shap_values

            # Save bar plot
            plt.figure()
            shap.summary_plot(shap_values, processed_X, plot_type="bar", show=False)
            plt.savefig(
                f"{output_dir}/{self.hardware}/bar_{target}_{self.model_name}.png",
                bbox_inches="tight",
                dpi=300,
            )
            plt.close()

            # Save beeswarm plot
            plt.figure()
            shap.summary_plot(shap_values, processed_X, show=False)
            plt.savefig(
                f"{output_dir}/{self.hardware}/{target}_{self.model_name}.png",
                bbox_inches="tight",
                dpi=300,
            )
            plt.close()

        return shap_values_dict

    def calculate_tunability(self, shap_values_dict):
        """
        Compute "tunability" per processed feature column for each metric.

        Definition
        ----------
        For a given feature, tunability = mean of positive SHAP values across samples.
        This reflects how much increasing the feature tends to improve the metric.

        Returns
        -------
        dict[str, pd.Series]
            Per-metric Series of tunability scores indexed by processed feature names.
        """
        processed_X = self._preprocess_data()
        tunability_scores = {}

        for target, shap_values in shap_values_dict.items():
            shap_df = pd.DataFrame(shap_values, columns=processed_X.columns)
            tunability = shap_df.apply(lambda x: x[x > 0].mean()).sort_values(
                ascending=False
            )
            tunability_scores[target] = tunability

        return tunability_scores

    # ===== New: Weighted scoring at the parameter level =====
    def compute_weighted_param_scores(self, tunability_scores, weights=None):
        """
        Combine tunability across multiple metrics using weights, then aggregate
        the feature-level scores back to base parameters.

        Example
        -------
        score(param) = 0.7 * tunability_tps(param) + 0.3 * tunability_gpu(param)

        For enum parameters, the one-hot feature scores are combined first, then
        summed into the base parameter.

        Parameters
        ----------
        tunability_scores : dict[str, pd.Series]
            Per-metric tunability, indexed by processed feature names.
        weights : dict[str, float] or None
            Mapping from metric name to weight. Defaults to {"tps_avg": 0.7, "gpu_avg": 0.3}.

        Returns
        -------
        pd.Series
            Parameter-level scores sorted descending.
        """
        if weights is None:
            weights = {"tps_avg": 0.7, "gpu_avg": 0.3}

        processed_X = self._preprocess_data()
        feature_index = processed_X.columns

        # Accumulate weighted feature-level scores across available metrics.
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
            raise ValueError(
                "No available metrics found for weighting. "
                f"Expected any of {list(weights.keys())}, got {list(tunability_scores.keys())}."
            )

        # Aggregate feature scores into parameter scores (sum over mapped columns).
        param_scores = {}
        for base_param, feat_cols in self.param_feature_map.items():
            if not feat_cols:
                param_scores[base_param] = 0.0
            else:
                param_scores[base_param] = float(
                    combined_feature_scores.reindex(feat_cols).fillna(0.0).sum()
                )

        param_scores = pd.Series(param_scores).sort_values(ascending=False)
        return param_scores

    # ===== New: Write ranked JSON with "rank" field =====
    def write_ranked_params_json(self, ranked_params: pd.Series, output_path: str):
        """
        Write a ranked parameters JSON by injecting a "rank" field into the
        original knobs schema and sorting by that rank.

        Parameters
        ----------
        ranked_params : pd.Series
            Descending scores indexed by base parameter names.
        output_path : str
            Destination JSON path.

        Returns
        -------
        str
            The output path for convenience.
        """
        original = self._load_params()

        # Build rank mapping (1-based).
        rank_map = {}
        rank_counter = 1
        for p in ranked_params.index:
            rank_map[p] = rank_counter
            rank_counter += 1

        # Any parameters not present in ranked_params (edge case) go to the end.
        for p in original.keys():
            if p not in rank_map:
                rank_map[p] = rank_counter
                rank_counter += 1

        # Sort by rank and emit a fresh dict with "rank" added.
        sorted_items = sorted(original.items(), key=lambda kv: rank_map[kv[0]])

        out_dict = {}
        for p, meta in sorted_items:
            meta_copy = dict(meta)  # avoid in-place mutation
            meta_copy["rank"] = int(rank_map[p])
            out_dict[p] = meta_copy

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(out_dict, f, ensure_ascii=False, indent=4)

        return output_path

    def save_metric_extrema_from_memory(self, model: str, output_path: str = None):
        """
        Compute min/max for each available metric directly from `self.y` and
        save to a JSON file.

        Parameters
        ----------
        model : str
            Model name to use in the default output path.
        output_path : str or None
            Custom destination. If None, defaults to "bounds/{hardware}/{model}.json".

        Returns
        -------
        str
            The output path for convenience.
        """
        if self.y is None:
            raise ValueError("Dataset not generated yet; call generate_dataset first.")

        extrema = {}
        metric_candidates = ["tps_avg", "gpu_avg"]

        for m in metric_candidates:
            if m in self.y.columns:
                s = pd.to_numeric(self.y[m], errors="coerce")
                extrema[m] = {
                    "min": float(np.nanmin(s.values)) if len(s) else None,
                    "max": float(np.nanmax(s.values)) if len(s) else None,
                }

        if not extrema:
            raise ValueError(
                f"No metric columns found in self.y (candidates: {metric_candidates})."
            )

        if output_path is None:
            output_path = f"bounds/{self.hardware}/{model}.json"

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(extrema, f, ensure_ascii=False, indent=4)

        return output_path


# Example usage
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Llama Configuration Optimizer")
    parser.add_argument(
        "--device",
        type=str,
        choices=["cpu", "gpu"],
        default="gpu",
        help="Processing device (cpu or gpu)",
    )
    parser.add_argument(
        "--hardware",
        type=str,
        choices=["rtx3060", "rtx4090", "m4", "orin"],
        default="rtx3060",
        help="Processing hardware",
    )
    parser.add_argument(
        "--model",
        type=str,
        choices=["qwen3-4b", "phimoe-mini"],
        default="phimoe-mini",
        help="Model family (e.g., qwen3-8b, phimoe-mini)",
    )
    parser.add_argument(
        "--quant", type=str, choices=["q4", "q8"], default="q8", help="Quantization tag"
    )
    args = parser.parse_args()

    model_name = f"{args.model}-{args.quant}"  # model identifier like "qwen3-4b-q8"

    optimizer = LlamaConfigOptimizer(
        params_path="knobs_files/knobs_raw.json",
        device=args.device,
        model=f"./../models/{model_name}.gguf",
        model_name=model_name,
        hardware=args.hardware,
    )

    # 1) Generate dataset
    X, y = optimizer.generate_dataset(n_samples=100)
    os.makedirs(f"shap_outputs/{args.hardware}", exist_ok=True)

    extrema_json_path = optimizer.save_metric_extrema_from_memory(model_name)
    print(f"Saved metric extrema JSON to: {extrema_json_path}")

    X.to_csv(f"shap_outputs/{args.hardware}/X_{model_name}.csv", index=False)
    y.to_csv(f"shap_outputs/{args.hardware}/y_{model_name}.csv", index=False)

    # 2) Train models
    optimizer.train_model()

    # 3) SHAP analysis
    shap_values_dict = optimizer.analyze_shap()

    # 4) Per-metric tunability (feature-level)
    tunability_scores = optimizer.calculate_tunability(shap_values_dict)

    # 5) Weighted aggregation to parameter-level scores and ranking
    weighted_param_scores = optimizer.compute_weighted_param_scores(
        tunability_scores, weights={"tps_avg": 0.7, "gpu_avg": 0.3}
    )

    print("\nWeighted Param Scores (desc):")
    print(weighted_param_scores)

    # 6) Write ranked JSON (injects `rank` and sorts)
    output_json = f"knobs_files/{args.hardware}/{model_name}.json"
    path = optimizer.write_ranked_params_json(weighted_param_scores, output_json)
    print(f"\nWrote ranked params JSON to: {path}")
