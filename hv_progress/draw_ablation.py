import json
import os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

# ==============================
# 基本配置（按需修改）
# ==============================
model = "phimoe-mini-q4"

# 4 个硬件（1x4 面板）
hardwares = ["rtx4090", "rtx3060", "m4", "orin"]
hardware_name = {
    "rtx4090": "RTX 4090",
    "rtx3060": "RTX 3060",
    "m4": "Apple M4",
    "orin": "Orin-Nano"
}

# 只画 LaTune 及其两个消融
methods = {
    "wo1": {"label": "LaTune-w/o-PS", "linestyle": "--",  "marker": "x", "color": "#8c564b"},
    "wo2": {"label": "LaTune-w/o-KT", "linestyle": "-.",  "marker": "*", "color": "#e377c2"},
    "latune": {"label": "LaTune",     "linestyle": "-",   "marker": "v", "color": "#9467bd"},
}

# 迭代信息
total_iters = 50
record_every = 5
x_points = np.arange(record_every, total_iters + 1, record_every)

# ==============================
# 全局绘图风格
# ==============================
plt.rcParams.update({
    "font.size": 25,
    "axes.labelsize": 25,
    "axes.titlesize": 25,
    "xtick.direction": "in",
    "ytick.direction": "in",
})

# 1x4 子图
fig, axs = plt.subplots(1, 4, figsize=(16, 3), dpi=150, sharex=False, sharey=False)
axs = np.array(axs).reshape(-1)

legend_handles, legend_labels = {}, {}

# ==============================
# 工具函数
# ==============================
def read_hv_series(hardware, model, method):
    fpath = f"{hardware}/{model}-{method}.json"
    if not os.path.exists(fpath):
        print(f"[WARN] 文件缺失：{fpath}")
        return None
    try:
        with open(fpath, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data
    except Exception as e:
        print(f"[ERROR] 读取失败：{fpath} → {e}")
        return None

def plot_stair(ax, x, y, style):
    ln = ax.step(
        x, y, where="post",
        linewidth=1.8,
        linestyle=style["linestyle"],
        color=style["color"],
        marker=style["marker"],
        markersize=7.5,
        label=style["label"]
    )
    ax.plot(x, y, linestyle="none",
            marker=style["marker"], markersize=4.5,
            color=style["color"])
    return ln[0]

# ==============================
# 主绘图循环
# ==============================
for idx, hardware in enumerate(hardwares):
    ax = axs[idx]
    ax.set_xlabel("Iterations", fontsize=24)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    for mkey, mstyle in methods.items():
        y = read_hv_series(hardware, model, mkey)
        if y is None:
            continue

        y = list(y)
        if len(y) < len(x_points):
            y = y + [y[-1]] * (len(x_points) - len(y))
        elif len(y) > len(x_points):
            y = y[:len(x_points)]

        h = plot_stair(ax, x_points, y, mstyle)
        legend_handles[mstyle["label"]] = h
        legend_labels[mstyle["label"]] = mstyle["label"]

    ax.set_title(hardware_name[hardware],fontweight="bold", fontsize=24)
    ax.set_xlabel("Iterations")

    # 只在第一个子图显示 y 轴标签
    if idx == 0:
        ax.set_ylabel("HV")


# ==============================
# 图例与保存
# ==============================
handles = list(legend_handles.values())
labels = list(legend_labels.values())

plt.subplots_adjust(top=0.82, wspace=0.25, hspace=0.3)

fig.legend(
    handles, labels,
    loc="upper center",
    ncol=3,
    frameon=False,
    bbox_to_anchor=(0.5, 1.1),
    handlelength=2.2,
    borderaxespad=-0.2,
    columnspacing=1.2
)

for ax in axs:
    ax.set_xlim(x_points[0], x_points[-1])

plt.savefig("hv_stair_ablation.pdf", bbox_inches="tight")
plt.show()
