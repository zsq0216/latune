#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
用法：
    python gen_table_cell.py

功能：
- 外层循环：methods = (Default, GA, CBO, scoot, latune)
- 中层：models = (qwen3-4b-q4, qwen3-4b-q8, phimoe-mini-q4, phimoe-mini-q8)
- 内层：folders = (rtx4090, rtx3060, m4, orin)
- 从 "{folder}/{model}-{method}.json" 读取最后一个数 ×100。
- 输出为 LaTeX 表格行，并在「每列的最大值」加粗（\textbf{}）
- 缺失值输出 N/A，不参与最大值比较。
"""

import os
import json

MODELS = ("qwen3-4b-q4", "qwen3-4b-q8", "phimoe-mini-q4", "phimoe-mini-q8")
METHODS = ("Default", "GA", "CBO", "scoot", "latune")
FOLDERS = ["rtx4090", "rtx3060", "m4", "orin"]

# 方法名映射表
METHOD_DISPLAY = {
    "CBO": "ResTune",
    "scoot": "SCOOT",
    "latune": "LaTune"
}

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
                    row_values.append(v * 100)
        values.append(row_values)

    # 计算每列的最大值（忽略 None）
    num_cols = len(MODELS) * len(FOLDERS)
    col_max = []
    for col in range(num_cols):
        col_vals = [values[row][col] for row in range(len(METHODS)) if values[row][col] is not None]
        col_max.append(max(col_vals) if col_vals else None)

    # 输出每行
    for row_idx, method in enumerate(METHODS):
        display_name = METHOD_DISPLAY.get(method, method)
        line_values = []
        for col_idx, v in enumerate(values[row_idx]):
            if v is None:
                line_values.append("N/A")
            elif col_max[col_idx] is not None and v == col_max[col_idx]:
                line_values.append(f"\\textbf{{{v:.2f}}}")
            else:
                line_values.append(f"{v:.2f}")

        line = f"{display_name} &" + " & ".join(line_values)
        # print(line)
        # 如需 LaTeX 换行可用：
        print(line + r" \\")
        # 或 print(line + r" \\ \hline")

    # # ===== 新增：Boost 行（LaTune ÷ Default 的倍数，后缀 x）=====
    # try:
    #     default_row = values[METHODS.index("Default")]
    #     latune_row = values[METHODS.index("latune")]
    # except ValueError:
    #     # 极端情况：方法列表被改动
    #     default_row = None
    #     latune_row = None

    # boost_values = []
    # if default_row is not None and latune_row is not None:
    #     for dv, lv in zip(default_row, latune_row):
    #         if dv is None or lv is None:
    #             boost_values.append("N/A")
    #         else:
    #             # 避免除以 0
    #             if dv == 0:
    #                 boost_values.append("N/A")
    #             else:
    #                 ratio = lv / dv
    #                 boost_values.append(f"{ratio:.2f}x")
    # else:
    #     boost_values = ["N/A"] * num_cols

    # boost_line = "Boost &" + " & ".join(boost_values)
    # print(boost_line + r" \\")  # 追加到表格最后一行


if __name__ == "__main__":
    main()
