#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch
import matplotlib.patheffects as pe
from matplotlib.lines import Line2D

RESOURCE_ORDER = ["low", "mid", "high"]
METHOD_ORDER = ["Default", "GA", "CBO", "scoot", "latune"]

METHOD_STYLES = {
    "Default": {"color": "#F7D58B", "hatch": ""},
    "GA": {"color": "#CAB2D6", "hatch": "//"},
    "CBO": {"color": "#9BC985", "hatch": "xx"},
    "scoot": {"color": "#7DAEE0", "hatch": ".."},
    "latune": {"color": "#635ADD", "hatch": "\\\\"},
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
    width = 0.5
    gap_within = 0
    gap_between_groups = 0.4

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
    # —— 统一放大全局字体（含坐标刻度、legend 等）——
    plt.rcParams.update({
        "font.size": 14,        # 基础字号
        "axes.titlesize": 16,   # 标题
        "axes.labelsize": 15,   # 坐标轴标题
        "legend.fontsize": 13,  # 图例
        "xtick.labelsize": 13,
        "ytick.labelsize": 13,
    })

    keys = [(r, m) for r in RESOURCE_ORDER for m in METHOD_ORDER]
    TPS_vals = [data_map.get(k, {}).get("TPS", None) for k in keys]
    VRAM_vals = [data_map.get(k, {}).get("VRAM", None) for k in keys]

    x, group_centers = build_positions()

    fig, ax1 = plt.subplots(figsize=(12, 6), dpi=150)
    ax2 = ax1.twinx()

    # === 左轴：TPS 柱状（不同方法用颜色+图案区分） ===
    bar_width = 0.5
    bar_rects = []  # 记录每根柱，便于加数值标签
    for i, y in enumerate(TPS_vals):
        res_idx = i // len(METHOD_ORDER)
        met_idx = i % len(METHOD_ORDER)
        method = METHOD_ORDER[met_idx]
        style = METHOD_STYLES[method]

        if y is None:
            ax1.scatter(x[i], 0, marker="x", s=70, color=style["color"], zorder=5)
        else:
            rect = ax1.bar(
                x[i],
                y,
                width=bar_width,
                color=style["color"],
                hatch=style["hatch"],
                edgecolor="black",
                linewidth=0.6,
                alpha=0.9,
            )[0]
            bar_rects.append((rect, y))

    ax1.set_ylabel("TPS")

    # === 右轴：VRAM 折线（统一颜色、显眼一些），每个 resource 分段绘制但只在第一段加 label 用于图例 ===
    n_met = len(METHOD_ORDER)
    vram_color = "red"  # 统一颜色
    outline = [pe.Stroke(linewidth=2.0, foreground="black"), pe.Normal()]  # 线的黑色描边

    vram_line_handles = []
    for gi, res in enumerate(RESOURCE_ORDER):
        start = gi * n_met
        end = start + n_met
        xs = x[start:end]
        ys = [np.nan if VRAM_vals[j] is None else VRAM_vals[j] for j in range(start, end)]

        line = ax2.plot(
            xs,
            ys,
            marker="s",                 # 方形点
            markersize=7,
            markerfacecolor=vram_color,
            markeredgecolor="black",    # 点的黑色描边
            markeredgewidth=1.2,
            linewidth=2.0,
            color=vram_color,
            solid_capstyle="round",
            label="VRAM" if gi == 0 else None,  # 只在第一段打标签（图例合并）
            zorder=4,
        )[0]
        # 给折线本体加黑色描边（更醒目）
        line.set_path_effects(outline)

        if gi == 0:
            vram_line_handles.append(line)

        # 缺失位置标 x（右轴），与主色一致、更醒目
        for j in range(start, end):
            if VRAM_vals[j] is None:
                ax2.scatter(
                    x[j],
                    0,
                    marker="x",
                    s=80,
                    color=vram_color,
                    linewidths=2.0,
                    zorder=5,
                    path_effects=outline,
                )

    ax2.set_ylabel("VRAM")

    # === X 轴：不显示方法名称，只保留组标签（low/mid/high） ===
    ax1.set_xticks([])
    ax1.tick_params(axis="x", which="both", bottom=False, top=False, labelbottom=False)
    for ci, res in zip(group_centers, RESOURCE_ORDER):
        ax1.text(
            ci, -0.06, res.upper(),
            ha="center", va="top", fontsize=13, fontweight="bold",
            transform=ax1.get_xaxis_transform()
        )

    ax1.grid(axis="y", linestyle="--", alpha=0.3)
    ax1.margins(x=0.02)
    # ax1.set_title(title or "TPS（柱，左轴） / VRAM（折线，右轴）")

    # === 柱顶数值标签 ===
    # 对每根有效柱在顶部标注数值，自动根据数据范围设置偏移
    if bar_rects:
        # 用 y 轴范围估一个合适的偏移
        ymax = max([y for (_, y) in bar_rects] + [0])
        offset = ymax * 0.015 if ymax > 0 else 0.05
        for rect, y in bar_rects:
            height = rect.get_height()
            ax1.text(
                rect.get_x() + rect.get_width() / 2.0,
                height + offset,
                f"{y:.2f}",               # 保留两位小数，可按需改成 {y:.1f}/{y:.0f}
                ha="center", va="bottom",
                fontsize=12
            )

    # === 统一图例：方法样例 + VRAM（右上角） ===
    method_patches = [
        Patch(facecolor=METHOD_STYLES[m]["color"], hatch=METHOD_STYLES[m]["hatch"],
              edgecolor="black", label=m.upper())
        for m in METHOD_ORDER
    ]
    combined_handles = method_patches + vram_line_handles
    ax1.legend(handles=combined_handles, loc="upper right", frameon=False, title="LEGEND")

    fig.tight_layout()
    if output:
        plt.savefig(output, bbox_inches="tight")
        print(f"[OK] 已保存图像到: {output}")
    else:
        plt.show()



def main():
    parser = argparse.ArgumentParser(description="绘制 TPS（柱状）+ VRAM（折线）复合图")

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
        plot(data_map, title=None, output=output)


if __name__ == "__main__":
    main()
