#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

RESOURCE_ORDER = ["low", "mid", "high"]
METHOD_ORDER = ["Default", "GA", "ResTune", "SCOOT", "LaTune"]

METHOD_STYLES = {
    "Default": {"color": "#F7D58B", "hatch": ""},
    "GA": {"color": "#CAB2D6", "hatch": "//"},
    "ResTune": {"color": "#9BC985", "hatch": "xx"},
    "SCOOT": {"color": "#7DAEE0", "hatch": ".."},
    "LaTune": {"color": "#DDA52D", "hatch": "--"},
}

MODEL_LIST = ["qwen3-4b-q4", "qwen3-4b-q8", "phimoe-mini-q4", "phimoe-mini-q8"]

# -------------------------------
# 字体与图形尺寸配置（论文优化）
# -------------------------------
FONT_CFG = {
    "font_size_base": 28,   # 全局默认字体
    "title_size": 32,       # 子图标题字体
    "label_size": 36,       # 坐标轴标签字体（纵坐标特别大）
    "tick_size": 30,        # 坐标刻度字体
    "legend_size": 30,      # 图例字体
    "group_label_size": 28, # 模型名字体
}

FIGSIZE = (30, 8)
DPI = 300

def load_data(path: Path):
    with path.open("r", encoding="utf-8") as f:
        arr = json.load(f)
    data = {}
    for row in arr:
        res = str(row.get("resource", "")).lower()
        met = str(row.get("method", ""))
        if res not in RESOURCE_ORDER or met not in METHOD_ORDER:
            continue
        tps = row.get("tps_avg", None)
        tps = float(tps) if isinstance(tps, (int, float)) else None
        data[(res, met)] = tps
    return data

def build_positions(n_models: int, n_methods: int):
    width = 0.30
    gap_between_models = 0.70
    x_positions, group_centers = [], []
    base = 0.0
    for _ in range(n_models):
        for mi in range(n_methods):
            x_positions.append(base + mi * width)
        start = base
        end = x_positions[-1]
        group_centers.append((start + end) / 2.0)
        base = x_positions[-1] + width + gap_between_models
    return np.array(x_positions), np.array(group_centers), width

def plot_subplot(ax, resource: str, model_data_map: dict, models_available: list):
    n_models = len(models_available)
    n_methods = len(METHOD_ORDER)
    if n_models == 0:
        ax.set_axis_off()
        ax.text(0.5, 0.5, f"No data for {resource.upper()}",
                ha="center", va="center", fontsize=FONT_CFG["title_size"])
        return

    x, group_centers, bar_width = build_positions(n_models, n_methods)

    x_pad = bar_width * 0.8
    ax.set_xlim(x.min() - x_pad, x.max() + x_pad)

    TPS_vals = []
    for model in models_available:
        mdata = model_data_map.get(model, {})
        for method in METHOD_ORDER:
            TPS_vals.append(mdata.get((resource, method), None))

    subplot_mins = []
    for i, y in enumerate(TPS_vals):
        method = METHOD_ORDER[i % n_methods]
        style = METHOD_STYLES[method]
        if y is None or y <= 1:
            ax.text(
                x[i], 0.05, "Error", rotation=90, ha="center", va="bottom",
                fontsize=max(18, int(FONT_CFG["tick_size"] * 0.9)),
                color="#d2691e", fontweight="bold",
                transform=ax.get_xaxis_transform()
            )
        else:
            subplot_mins.append(float(y))
            ax.bar(
                x[i], y, width=bar_width,
                color=style["color"], hatch=style["hatch"],
                edgecolor="black", linewidth=0.7, alpha=0.9,
            )

    # 标题与纵坐标标签
    ax.set_ylabel("TPS", fontsize=FONT_CFG["label_size"])
    ax.set_title(resource.upper(), fontsize=FONT_CFG["title_size"], fontweight="bold")

    # 横坐标模型名（斜着放）
    ax.set_xticks(group_centers)
    ax.set_xticklabels(models_available, rotation=30, ha="right",
                       fontsize=FONT_CFG["group_label_size"])

    ax.tick_params(axis="y", labelsize=FONT_CFG["tick_size"])

    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.margins(x=0.02)

    if len(subplot_mins) > 0:
        y_min = 0.9 * min(subplot_mins)
        ax.set_ylim(bottom=y_min)

def main():
    parser = argparse.ArgumentParser(description="三个子图横排展示TPS")
    args = parser.parse_args()

    plt.rcParams.update({
        "font.size": FONT_CFG["font_size_base"],
        "axes.titlesize": FONT_CFG["title_size"],
        "axes.labelsize": FONT_CFG["label_size"],
        "legend.fontsize": FONT_CFG["legend_size"],
        "xtick.labelsize": FONT_CFG["tick_size"],
        "ytick.labelsize": FONT_CFG["tick_size"],
    })

    model_data_map = {}
    models_available = []
    for model in MODEL_LIST:
        in_path = Path(f"{model}.json")
        if not in_path.exists():
            print(f"[WARN] 未找到输入文件：{in_path}，跳过该模型")
            continue
        try:
            model_data_map[model] = load_data(in_path)
            models_available.append(model)
        except Exception as e:
            print(f"[ERROR] 读取失败：{in_path} -> {e}")

    fig, axes = plt.subplots(1, 3, figsize=FIGSIZE, dpi=DPI, sharex=False, sharey=False)
    if not isinstance(axes, np.ndarray):
        axes = np.array([axes])

    for ax, res in zip(axes, RESOURCE_ORDER):
        plot_subplot(ax, res, model_data_map, models_available)

    method_patches = [
        Patch(facecolor=METHOD_STYLES[m]["color"], hatch=METHOD_STYLES[m]["hatch"],
              edgecolor="black", label=m)
        for m in METHOD_ORDER
    ]
    fig.legend(
        handles=method_patches,
        loc='upper center',
        bbox_to_anchor=(0.5, 1.05),
        ncol=len(METHOD_ORDER),
        frameon=False
    )

    plt.tight_layout()
    plt.subplots_adjust(top=0.88, wspace=0.25)

    output_file = "adaptive-by-resource.pdf"
    plt.savefig(output_file, bbox_inches="tight")
    print(f"[OK] 已保存图像到: {output_file}")

if __name__ == "__main__":
    main()
