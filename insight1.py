import json
from pathlib import Path
from llama_executor import LlamaExecutor

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":
    # 1) Parameter type definitions
    param_types_instance = {
        'gpu-layers': 'integer',
        'ctx-size': 'integer',
        'no-kv-offload': 'boolean',
        'flash-attn': 'boolean',
        'parallel': 'integer',
        "no-cont-batching": "boolean",
        "thread": "integer",
        "ubatch-size": "integer"
    }

    # 2) Configuration list
    config_options = {
        "default": {"gpu-layers":32},
        "gpu-layers": {"gpu-layers": 100},
        # "ctx-size": {"gpu-layers":32, "ctx-size": 8192},
        "no-kv-offload": {"gpu-layers":32, "no-kv-offload": True},
        "flash-attn": {"gpu-layers":32, "flash-attn": True},
        "parallel": {"gpu-layers":32, "parallel": 8},
        "no-cont-batching": {"gpu-layers":32, "no-cont-batching": True},
        "ubatch-size": {"gpu-layers":32, "ubatch-size": 4096},
    }


    # 3) Executor initialization
    executor = LlamaExecutor(
        param_types=param_types_instance,
        model_path="./../models/phimoe-mini-q4.gguf",
        device="gpu",
    )

    # 4) Run all configurations
    results = []
    for name, cfg in config_options.items():
        result = executor.run_server_performance_test(cfg)
        result.update({"config_name": name})
        result.update(cfg)
        results.append(result)
        print(result)

    # 5) Convert to DataFrame
    df = pd.DataFrame(results)
    required_cols = {"config_name", "tps_avg", "gpu_avg"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing required fields: {missing}")

    if "default" not in set(df["config_name"]):
        raise ValueError("No 'default' configuration found.")

    base = df.loc[df["config_name"] == "default"].iloc[0]
    tps_base = base["tps_avg"]
    gpu_base = base["gpu_avg"]

    # Calculate relative change
    df["tps_delta_pct"] = (df["tps_avg"] - tps_base) / tps_base
    df["gpu_delta_pct"] = (df["gpu_avg"] - gpu_base) / gpu_base

    # Sort by absolute change before output
    df_sorted_tps = (
        df[df["config_name"] != "default"]
        .sort_values("tps_delta_pct", key=lambda s: s.abs(), ascending=True)
    )
    df_sorted_gpu = (
        df[df["config_name"] != "default"]
        .sort_values("gpu_delta_pct", key=lambda s: s.abs(), ascending=True)
    )

    # 6) Print sorted summary
    def fmt_pct(x):
        return f"{x*100:.2f}%"

    print("\n=== Sorted Summary by |TPS Change| ===")
    print(
        df_sorted_tps[["config_name", "tps_avg", "gpu_avg", "tps_delta_pct", "gpu_delta_pct"]]
        .assign(
            tps_delta_pct=lambda d: d["tps_delta_pct"].map(fmt_pct),
            gpu_delta_pct=lambda d: d["gpu_delta_pct"].map(fmt_pct),
        )
        .to_string(index=False)
    )

    # 7) Tornado Plot for TPS
    fig_tps, ax1 = plt.subplots(figsize=(7, 5))
    ax1.barh(
        df_sorted_tps["config_name"],
        df_sorted_tps["tps_delta_pct"],
        color="steelblue",
        edgecolor="black"
    )
    ax1.set_title("(a) TPS Relative Change (vs. default)", fontsize=13)
    ax1.set_xlabel("Relative Change (%)", fontsize=11)
    ax1.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x*100:.0f}%"))
    ax1.axvline(0, color="black", linewidth=1)
    ax1.grid(axis="x", linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig("tornado_tps.pdf", format="pdf", dpi=300, bbox_inches="tight")
    plt.close(fig_tps)

    # 8) Tornado Plot for GPU Utilization
    fig_gpu, ax2 = plt.subplots(figsize=(7, 5))
    ax2.barh(
        df_sorted_gpu["config_name"],
        df_sorted_gpu["gpu_delta_pct"],
        color="indianred",
        edgecolor="black"
    )
    ax2.set_title("(b) GPU Utilization Relative Change (vs. default)", fontsize=13)
    ax2.set_xlabel("Relative Change (%)", fontsize=11)
    ax2.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x*100:.0f}%"))
    ax2.axvline(0, color="black", linewidth=1)
    ax2.grid(axis="x", linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig("tornado_gpu.pdf", format="pdf", dpi=300, bbox_inches="tight")
    plt.close(fig_gpu)

    print("\n✅ Tornado charts saved as:")
    print(" - tornado_tps.pdf")
    print(" - tornado_gpu.pdf")
