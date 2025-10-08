#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
用法：
    python gen_table_cell_gap.py

功能：
- 外层：methods = (Default, GA, CBO, scoot, latune)
- 中层：models = (qwen3-4b-q4, qwen3-4b-q8, phimoe-mini-q4, phimoe-mini-q8)
- 内层：folders = (rtx4090, rtx3060, m4, orin)
- 从 "{folder}/{model}-{method}.json" 读取最后一个数 ×100。
- 仅输出：每一列（模型×硬件）的 “最大值 与 第二大不同值” 之差（top1 - top2）。
  - 若该列不足两个不同的有效数值，则输出 N/A。
"""

import os
import json

MODELS = ("qwen3-4b-q4", "qwen3-4b-q8", "phimoe-mini-q4", "phimoe-mini-q8")
METHODS = ("Default", "GA", "CBO", "scoot", "latune")
FOLDERS = ["rtx4090", "rtx3060", "m4", "orin"]

# 方法名映射表（此处不再逐行输出，仅保留以备扩展）
METHOD_DISPLAY = {
    "CBO": "ResTune",
    "scoot": "SCOOT",
    "latune": "LaTune"
}

EPS = 1e-9  # 浮点比较公差

def read_last_value(path):
    """从 json 文件读取最后一个数；返回 None 表示失败。"""
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, list) or len(data) == 0:
            return None
        return data[-1]
    except Exception:
        return None


def main():
    # 读取所有数值到矩阵: methods × (models × folders)
    values = []  # list of list
    for method in METHODS:
        row_values = []
        for model in MODELS:
            for folder in FOLDERS:
                filename = f"{model}-{method}.json"
                fullpath = os.path.join(folder, filename)
                v = read_last_value(fullpath)
                if v is None:
                    row_values.append(None)
                else:
                    row_values.append(v * 100)  # 放大为百分制
        values.append(row_values)

    # 计算每列的最大与第二大“不同数值”（忽略 None）
    num_cols = len(MODELS) * len(FOLDERS)
    gaps = []  # 每列 top1 - top2；不足两个不同数值则为 None
    for col in range(num_cols):
        col_vals = [values[row][col] for row in range(len(METHODS)) if values[row][col] is not None]
        # 去重（按浮点公差）后降序
        # 简单做法：直接用 set 去重；若需要更严格的公差聚类，可扩展
        uniq_sorted = sorted(set(col_vals), reverse=True)
        if len(uniq_sorted) >= 2:
            top1, top2 = uniq_sorted[0], uniq_sorted[1]
            gaps.append((top1 - top2)/top2 * 100)  # 计算百分比差距
        else:
            gaps.append(None)

    # 以 LaTeX 表格行的形式输出：Top-2 Gap 行
    line_values = []
    for g in gaps:
        if g is None:
            line_values.append("N/A")
        else:
            line_values.append(f"{g:.2f}")
    line = "Top-2 Gap & " + " & ".join(line_values)
    print(line + r" \\")


if __name__ == "__main__":
    main()
