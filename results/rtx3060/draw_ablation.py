#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

# === 只保留三个资源档位 & 两个方法 ===
RESOURCE_ORDER = ["low", "mid", "high"]
METHOD_ORDER = ["LaTune","LaTune-w/o-TS"]  

METHOD_STYLES = {
    "LaTune-w/o-TS": {"color": "#9AA8B3", "hatch": "\\\\"},
    "LaTune": {"color": "#267785", "hatch": ""},
}

def load_data(path: Path):
    """
    读取 json 文件，按 (resource, method) 索引记录，只保留 TPS。
    仅接收 RESOURCE_ORDER × METHOD_ORDER 的组合。
    """
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

def build_positions(n_res: int, n_met: int, bar_width: float = 0.32,
                    group_gap: float = 0.8, intra_gap: float = 0.06):
    """
    分组对齐版：
    - 先确定每个资源组的“组中心”（等间距）。
    - 组内每个 method 围绕组中心对称偏移，确保视觉居中不偏。
    """
    # 每个资源组的中心位置（等距摆放）
    group_span = n_met * bar_width + (n_met - 1) * intra_gap + group_gap
    group_centers = np.arange(n_res) * group_span

    # 组内每根柱子的偏移（以组中心为 0 对称分布）
    offsets = (np.arange(n_met) - (n_met - 1) / 2.0) * (bar_width + intra_gap)

    # 展开成 [low组两个x, mid组两个x, high组两个x]
    x_positions = np.concatenate([group_centers[g] + offsets for g in range(n_res)])

    return x_positions, group_centers


def plot_one_axes(ax, data_map, title=None):
    keys = [(r, m) for r in RESOURCE_ORDER for m in METHOD_ORDER]
    TPS_vals = [data_map.get(k, {}).get("TPS", None) for k in keys]

    n_res = len(RESOURCE_ORDER)
    n_met = len(METHOD_ORDER)
    bar_width = 0.32

    x, group_centers = build_positions(n_res, n_met, bar_width=bar_width,
                                       group_gap=0.2, intra_gap=0)
    # 画柱
    for i, y in enumerate(TPS_vals):
        method = METHOD_ORDER[i % n_met]
        style = METHOD_STYLES[method]
        if y is None or y <= 1:
            ax.text(x[i], 0.05, "Error", rotation=90, ha="center",
                    va="bottom", fontsize=16, color="#d2691e", fontweight="bold")
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

    # y 轴与标题
    ax.set_ylabel("TPS")
    if title:
        ax.set_title(title, pad=8, fontweight="bold")

    # X 轴组标签（LOW / MID / HIGH）
    ax.set_xticks([])
    ax.tick_params(axis="x", bottom=False, top=False)
    for ci, res in zip(group_centers, RESOURCE_ORDER):
        ax.text(
            ci, -0.10, res.upper(),
            ha="center", va="top", fontsize=16, fontweight="bold",
            transform=ax.get_xaxis_transform()
        )

    # 让两侧留一点边距，避免第一/最后一组被裁
    half_span = (n_met * bar_width + (n_met - 1) * 0.08) / 2.0
    ax.set_xlim(group_centers[0] - half_span - 0.2, group_centers[-1] + half_span + 0.2)

    ax.grid(axis="y", linestyle="--", alpha=0.3)
    ax.margins(y=0.05)

def main():
    # 统一样式调小一些以适配 1×4 子图
    plt.rcParams.update({
        "font.size": 16,
        "axes.titlesize": 16,
        "axes.labelsize": 16,
        "legend.fontsize": 20,
        "xtick.labelsize": 16,
        "ytick.labelsize": 16,
    })

    model_list = ["qwen3-4b-q4", "qwen3-4b-q8", "phimoe-mini-q4", "phimoe-mini-q8"]

    # 创建 1×4 子图
    fig, axes = plt.subplots(1, 4, figsize=(12, 4.5), dpi=150, sharey=True)
    if not isinstance(axes, np.ndarray):
        axes = np.array([axes])

    # 每个子图读取对应 json 并绘制
    for ax, model in zip(axes, model_list):
        in_path = Path(f"{model}.json")
        if not in_path.exists():
            raise FileNotFoundError(f"未找到输入文件：{in_path}")
        data_map = load_data(in_path)
        plot_one_axes(ax, data_map, title=model)

    # 统一图例（放在右上角外侧或底部，可按需要调整 loc/bbox_to_anchor）
    method_patches = [
        Patch(facecolor=METHOD_STYLES[m]["color"], hatch=METHOD_STYLES[m]["hatch"],
              edgecolor="black", label=m)
        for m in METHOD_ORDER
    ]
    fig.legend(handles=method_patches, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.06))

    fig.tight_layout()
    plt.subplots_adjust(top=0.83)  # 给顶部图例留空间
    output_file = "adaptive-ablation.pdf"
    plt.savefig(output_file, bbox_inches="tight")
    print(f"[OK] 已保存对比图到: {output_file}")

if __name__ == "__main__":
    main()
