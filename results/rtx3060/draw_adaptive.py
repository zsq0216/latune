#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

RESOURCE_ORDER = ["low", "mid", "high"]
METHOD_ORDER = ["Default", "GA", "ResTune", "SCOOT", "LaTune-w/o-g", "LaTune"]

METHOD_STYLES = {
    "Default": {"color": "#F7D58B", "hatch": ""},
    "GA": {"color": "#CAB2D6", "hatch": "//"},
    "ResTune": {"color": "#9BC985", "hatch": "xx"},
    "SCOOT": {"color": "#7DAEE0", "hatch": ".."},
    "LaTune-w/o-g": {"color": "#736DC6", "hatch": "\\\\"},
    "LaTune": {"color": "#DDA52D", "hatch": "-"},
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

def plot(data_map, title=None, output=None):
    plt.rcParams.update({
        "font.size": 25,
        "axes.titlesize": 25,
        "axes.labelsize": 25,
        "legend.fontsize": 22,
        "xtick.labelsize": 25,
        "ytick.labelsize": 25,
    })

    keys = [(r, m) for r in RESOURCE_ORDER for m in METHOD_ORDER]
    TPS_vals = [data_map.get(k, {}).get("TPS", None) for k in keys]
    x, group_centers = build_positions()

    fig, ax = plt.subplots(figsize=(12, 6), dpi=150)

    # === 柱状图（TPS）===
    bar_width = 0.3
    for i, y in enumerate(TPS_vals):
        method = METHOD_ORDER[i % len(METHOD_ORDER)]
        style = METHOD_STYLES[method]
        if y is None or y <= 1:
            # ax.scatter(x[i], 0, marker="x", s=70, color=style["color"], zorder=5)
            ax.text(x[i], 0.05, "Error", rotation=90, ha="center",
                    va="bottom", fontsize=18, color="#d2691e",fontweight="bold")
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

    ax.set_ylabel("TPS")

    # === X 轴组标签（LOW/MID/HIGH）===
    ax.set_xticks([])
    ax.tick_params(axis="x", bottom=False, top=False)
    for ci, res in zip(group_centers, RESOURCE_ORDER):
        ax.text(
            ci, -0.06, res.upper(),
            ha="center", va="top", fontsize=20, fontweight="bold",
            transform=ax.get_xaxis_transform()
        )

    ax.grid(axis="y", linestyle="--", alpha=0.3)
    ax.margins(x=0.02)

    # === 图例 ===
    method_patches = [
        Patch(facecolor=METHOD_STYLES[m]["color"], hatch=METHOD_STYLES[m]["hatch"],
              edgecolor="black", label=m.upper())
        for m in METHOD_ORDER
    ]
    ax.legend(handles=method_patches, loc="upper right", frameon=False)

    fig.tight_layout()
    if output:
        plt.savefig(output, bbox_inches="tight")
        print(f"[OK] 已保存图像到: {output}")
    else:
        plt.show()

def main():
    parser = argparse.ArgumentParser(description="绘制 TPS 柱状图（按 resource×method 分组）")
    args = parser.parse_args()

    model_list = ["qwen3-4b-q4", "qwen3-4b-q8", "phimoe-mini-q4", "phimoe-mini-q8"]
    for model in model_list:
        input_file = f"{model}.json"
        output_file = f"adaptive-{model}.pdf"

        in_path = Path(input_file)
        if not in_path.exists():
            raise FileNotFoundError(f"未找到输入文件：{in_path}")

        data_map = load_data(in_path)
        plot(data_map, title=None, output=output_file)

if __name__ == "__main__":
    main()
