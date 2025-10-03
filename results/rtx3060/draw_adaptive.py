#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

RESOURCE_ORDER = ["low", "mid", "high"]
METHOD_ORDER = ["Default", "GA", "CBO", "scoot", "latune"]

METHOD_STYLES = {
    "Default": {"color": "tab:orange", "hatch": ""},
    "GA": {"color": "tab:blue", "hatch": "//"},
    "CBO": {"color": "tab:green", "hatch": "xx"},
    "scoot": {"color": "tab:red", "hatch": ".."},
    "latune": {"color": "tab:purple", "hatch": "\\\\"},
}

def load_data(path: Path):
    """读取 json 文件，按 (resource, method) 索引记录"""
    with path.open("r", encoding="utf-8") as f:
        arr = json.load(f)

    data = {}
    for row in arr:
        res = str(row.get("resource", "")).lower()
        met = str(row.get("method", ""))
        if res not in RESOURCE_ORDER or met not in METHOD_ORDER:
            continue
        tps = row.get("tps_avg", None)
        vram = row.get("gpu_avg", None)
        # 强制转为 float
        if isinstance(tps, (int, float)):
            tps = float(tps)
        else:
            tps = None
        if isinstance(vram, (int, float)):
            vram = float(vram)
        else:
            vram = None
        data[(res, met)] = {"TPS": tps, "VRAM": vram}
    return data


def build_positions():
    """生成每个 (resource, method) 的横坐标位置"""
    n_res = len(RESOURCE_ORDER)
    n_met = len(METHOD_ORDER)
    width = 0.8
    gap_within = 0
    gap_between_groups = 0.5

    x_positions = []
    base = 0.0
    group_centers = []

    for gi in range(n_res):
        for mi in range(n_met):
            x_positions.append(base + mi * (width + gap_within))
        start = base
        end = x_positions[-1]
        group_centers.append((start + end) / 2.0)
        base = x_positions[-1] + width + gap_between_groups

    return np.array(x_positions),  np.array(group_centers)

def plot(data_map, title=None, output=None):
    keys = [(r, m) for r in RESOURCE_ORDER for m in METHOD_ORDER]
    TPS_vals = [data_map.get(k, {}).get("TPS", None) for k in keys]
    VRAM_vals = [data_map.get(k, {}).get("VRAM", None) for k in keys]

    x, group_centers = build_positions()

    fig, ax1 = plt.subplots(figsize=(12, 6), dpi=150)
    ax2 = ax1.twinx()

    # === 左轴：TPS 柱状（不同方法用颜色+图案区分） ===
    bar_width = 0.8
    for i, y in enumerate(TPS_vals):
        res_idx = i // len(METHOD_ORDER)
        met_idx = i % len(METHOD_ORDER)
        method = METHOD_ORDER[met_idx]
        style = METHOD_STYLES[method]

        if y is None:
            # 缺失：在基线处画 x
            ax1.scatter(x[i], 0, marker="x", s=60, color=style["color"], zorder=5)
        else:
            ax1.bar(
                x[i],
                y,
                width=bar_width,
                color=style["color"],
                hatch=style["hatch"],
                edgecolor="black",
                linewidth=0.4,
                alpha=0.9,
            )

    ax1.set_ylabel("TPS")

    # === 右轴：VRAM 折线（按 resource 分段绘制，避免跨组连线） ===
    n_met = len(METHOD_ORDER)
    for gi, res in enumerate(RESOURCE_ORDER):
        start = gi * n_met
        end = start + n_met
        xs = x[start:end]
        ys = [np.nan if VRAM_vals[j] is None else VRAM_vals[j] for j in range(start, end)]
        ax2.plot(xs, ys, marker="o", linewidth=2, label=f"VRAM-{res}")

        # 缺失位置标 x（右轴）
        for j in range(start, end):
            if VRAM_vals[j] is None:
                ax2.scatter(x[j], 0, marker="x", s=60, color="tab:blue", zorder=6)

    ax2.set_ylabel("VRAM")

    # === X 轴：不显示方法名称，只保留组标签（low/mid/high） ===
    # ax1.set_xticks(ticks, [""] * len(ticks))
    ax1.set_xticks([])  
    ax1.tick_params(axis="x", which="both", bottom=False, top=False, labelbottom=False)
    for ci, res in zip(group_centers, RESOURCE_ORDER):
        ax1.text(
            ci, -0.06, res.upper(),
            ha="center", va="top", fontsize=11, fontweight="bold",
            transform=ax1.get_xaxis_transform()
        )

    ax1.grid(axis="y", linestyle="--", alpha=0.3)
    ax1.margins(x=0.02)
    ax1.set_title(title or "TPS（柱，左轴） / VRAM（折线，右轴）")

    # === 图例：方法样例 + VRAM 分段 ===
    method_patches = [
        Patch(facecolor=METHOD_STYLES[m]["color"], hatch=METHOD_STYLES[m]["hatch"],
              edgecolor="black", label=m.upper())
        for m in METHOD_ORDER
    ]
    # VRAM legend 使用 ax2 的线条
    h2, l2 = ax2.get_legend_handles_labels()
    legend1 = ax1.legend(handles=method_patches, title="METHOD", loc="upper left", frameon=False)
    ax1.add_artist(legend1)
    ax1.legend(h2, l2, title="VRAM (by RESOURCE)", loc="upper right", frameon=False)


    fig.tight_layout()
    if output:
        plt.savefig(output, bbox_inches="tight")
        print(f"[OK] 已保存图像到: {output}")
    else:
        plt.show()


def main():
    parser = argparse.ArgumentParser(description="绘制 TPS（柱状）+ VRAM（折线）复合图")
    # model = "qwen3-4b-q4"
    # model = "qwen3-4b-q8"
    # model = "phimoe-mini-q4"
    # model = "phimoe-mini-q8"
    model_list = ["qwen3-4b-q4","qwen3-4b-q8", "phimoe-mini-q4", "phimoe-mini-q8"]
    for model in model_list:
        input = f"{model}.json"
        output = f"adaptive-{model}.pdf"

        args = parser.parse_args()

        in_path = Path(input)
        title = Path(input).stem

        if not in_path.exists():
            raise FileNotFoundError(f"未找到输入文件：{in_path}")

        data_map = load_data(in_path)
        plot(data_map, title=f"{title} — TPS & VRAM by Resource/Method", output=output)


if __name__ == "__main__":
    main()
