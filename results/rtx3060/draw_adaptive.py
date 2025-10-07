#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')  # 必须在 import pyplot 之前设置 防止启动GUI
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch


RESOURCE_ORDER = ["low", "mid", "high"]
METHOD_ORDER = ["Default", "GA", "ResTune", "SCOOT", 
                # "LaTune-w/o-g", 
                "LaTune"]

METHOD_STYLES = {
    "Default": {"color": "#F7D58B", "hatch": ""},
    "GA": {"color": "#CAB2D6", "hatch": "//"},
    "ResTune": {"color": "#9BC985", "hatch": "xx"},
    "SCOOT": {"color": "#7DAEE0", "hatch": ".."},
    # "LaTune-w/o-g": {"color": "#736DC6", "hatch": "\\\\"},
    "LaTune": {"color": "#DDA52D", "hatch": "--"},
}

def load_data(path: Path):
    """读取 json 文件，按 (resource, method) 索引记录，只保留 TPS"""
    with path.open("r", encoding="utf-8") as f:
        arr = json.load(f)

    data = {}
    for row in arr:
        res = str(row.get("resource", "")).lower()
        met = str(row.get("method", ""))
        if res not in RESOURCE_ORDER or met not in METHOD_ORDER:
            continue

        tps = row.get("tps_avg", None)
        if isinstance(tps, (int, float)):
            tps = float(tps)
        else:
            tps = None

        data[(res, met)] = {"TPS": tps}
    return data

def build_positions():
    """生成每个 (resource, method) 的横坐标位置"""
    n_res = len(RESOURCE_ORDER)
    n_met = len(METHOD_ORDER)
    width = 0.3
    gap_between_groups = 0.3

    x_positions, group_centers = [], []
    base = 0.0

    for gi in range(n_res):
        for mi in range(n_met):
            x_positions.append(base + mi * width)
        start = base
        end = x_positions[-1]
        group_centers.append((start + end) / 2.0)
        base = x_positions[-1] + width + gap_between_groups

    return np.array(x_positions), np.array(group_centers)

def plot_subplot(ax, data_map, model_name):
    """在指定的子图上绘制数据"""
    keys = [(r, m) for r in RESOURCE_ORDER for m in METHOD_ORDER]
    TPS_vals = [data_map.get(k, {}).get("TPS", None) for k in keys]
    x, group_centers = build_positions()

    # === 柱状图（TPS）===
    bar_width = 0.3
    for i, y in enumerate(TPS_vals):
        method = METHOD_ORDER[i % len(METHOD_ORDER)]
        style = METHOD_STYLES[method]
        if y is None or y <= 1:
            ax.text(x[i], 0.05, "Error", rotation=90, ha="center",
                    va="bottom", fontsize=24, color="#d2691e", fontweight="bold")
        else:
            ax.bar(
                x[i],
                y,
                width=bar_width,
                color=style["color"],
                hatch=style["hatch"],
                edgecolor="black",
                linewidth=0.6,
                alpha=0.9,
            )

    ax.set_ylabel("TPS", fontsize=24)
    ax.set_title(model_name, fontsize=28, fontweight="bold")

    # === X 轴组标签（LOW/MID/HIGH）===
    ax.set_xticks([])
    ax.tick_params(axis="x", bottom=False, top=False)
    for ci, res in zip(group_centers, RESOURCE_ORDER):
        ax.text(
            ci, -0.06, res.upper(),
            ha="center", va="top", fontsize=24,
            transform=ax.get_xaxis_transform()
        )

    ax.grid(axis="y", linestyle="--", alpha=0.3)
    ax.margins(x=0.02)

def main():
    parser = argparse.ArgumentParser(description="绘制 TPS 柱状图（按 resource×method 分组）")
    args = parser.parse_args()

    # 设置全局字体大小
    plt.rcParams.update({
        "font.size": 24,
        "axes.titlesize": 28,
        "axes.labelsize": 24,
        "legend.fontsize": 24,
        "xtick.labelsize": 24,
        "ytick.labelsize": 24,
    })

    model_list = ["qwen3-4b-q4", "qwen3-4b-q8", "phimoe-mini-q4", "phimoe-mini-q8"]
    
    # 创建2x2的子图
    fig, axes = plt.subplots(2, 2, figsize=(16, 10), dpi=300)
    axes = axes.flatten()  # 将2D数组展平为1D以便于迭代
    
    # 绘制每个子图
    for i, model in enumerate(model_list):
        input_file = f"{model}.json"
        in_path = Path(input_file)
        if not in_path.exists():
            print(f"警告: 未找到输入文件：{in_path}，跳过该模型")
            continue
            
        data_map = load_data(in_path)
        plot_subplot(axes[i], data_map, model)
    
    # === 创建图例（放在整个图的顶部）===
    method_patches = [
        Patch(facecolor=METHOD_STYLES[m]["color"], hatch=METHOD_STYLES[m]["hatch"],
              edgecolor="black", label=m)
        for m in METHOD_ORDER
    ]
    fig.legend(handles=method_patches, 
               loc='upper center', 
               bbox_to_anchor=(0.5, 1.0),
               ncol=len(METHOD_ORDER),
               frameon=False)

    # 调整布局，为顶部图例留出空间
    plt.tight_layout()
    plt.subplots_adjust(top=0.88)  # 为顶部图例留出空间
    
    # 保存合并后的图像
    output_file = "adaptive-combined.pdf"
    plt.savefig(output_file, bbox_inches="tight")
    print(f"[OK] 已保存合并图像到: {output_file}")

if __name__ == "__main__":
    main()