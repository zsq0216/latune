import json
import os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

# ==============================
# 基本配置（按需修改）
# ==============================
# base_dir = Path("./data")   # 根目录，形如 data/{hardware}/{model}-{method}.json
model = "phimoe-mini-q4"   # 固定的模型名（请按实际修改）

# 4 个硬件（对应 2x2 面板）
hardwares = ["rtx4090", "rtx3060", "m4", "orin"]

# 5 个方法（集中定义风格，便于统一修改）
# 线型/颜色/marker 在此集中定义，后续统一循环绘图
methods = {
    "Default": {"label": "Default", "linestyle": "-",  "marker": "o", "color": "#1f77b4"},
    "GA": {"label": "GA", "linestyle": "--", "marker": "s", "color": "#ff7f0e"},
    "CBO": {"label": "CBO", "linestyle": "-.", "marker": "D", "color": "#2ca02c"},
    "scoot": {"label": "scoot", "linestyle": ":",  "marker": "^", "color": "#d62728"},
    "latune": {"label": "latune", "linestyle": "-",  "marker": "v", "color": "#9467bd"},
}

# 迭代信息：文件记录每 5 次迭代的 HV，一共 50 次迭代 → 10 个点
total_iters = 50
record_every = 5
x_points = np.arange(record_every, total_iters + 1, record_every)  # [5, 10, ..., 50]

# ==============================
# 画布与全局风格
# ==============================
plt.rcParams.update({
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 12,
    "xtick.direction": "in",
    "ytick.direction": "in",
})

fig, axs = plt.subplots(2, 2, figsize=(10, 7), dpi=150, sharex=True, sharey=True)
axs = axs.reshape(2, 2)

# 统一图例的句柄收集（保证所有面板一致）
legend_handles = {}
legend_labels = {}

# ==============================
# 工具函数：读取并绘制阶梯曲线
# ==============================
def read_hv_series(hardware, model, method):
    """
    从 {base_dir}/{hardware}/{model}-{method}.json 读取 HV 序列。
    返回 list[float]。若文件缺失则返回 None。
    """
    fpath = f"{hardware}/{model}-{method}.json"
    if not os.path.exists(fpath):
        print(f"[WARN] 文件缺失：{fpath}")
        return None
    try:
        with open(fpath, "r", encoding="utf-8") as f:
            data = json.load(f)
        # 期望 data 为 10 个浮点数（每 5 次迭代一个）
        return data
    except Exception as e:
        print(f"[ERROR] 读取失败：{fpath} → {e}")
        return None

def plot_stair(ax, x, y, style):
    """
    使用阶梯式绘图（台阶收敛可视化）
    - where='post' 让台阶在当前迭代维持到下一采样点
    - 同时在采样点位置添加 marker，便于读数
    """
    ln = ax.step(x, y, where="post",
                 linewidth=1.8,
                 linestyle=style["linestyle"],
                 color=style["color"],
                 label=style["label"])
    ax.plot(x, y, linestyle="none",
            marker=style["marker"], markersize=4.5,
            color=style["color"])
    return ln[0]  # step 返回一个 Line2D 列表，取第一个

# ==============================
# 主循环：每个硬件一个面板，面板内画所有方法
# ==============================
panel_tags = ["(a)", "(b)", "(c)", "(d)"]

for idx, hardware in enumerate(hardwares):
    r, c = divmod(idx, 2)
    ax = axs[r, c]

    # 去除上/右边框（论文风格）
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # 读取并绘制所有方法
    for mkey, mstyle in methods.items():
        y = read_hv_series(hardware, model, mkey)
        if y is None:
            continue

        # 容错：若长度与 x_points 不一致，做裁剪/填充（以免报错）
        y = list(y)
        if len(y) < len(x_points):
            # 简单填充（用最后一个值补齐），也可按需改为插值
            y = y + [y[-1]] * (len(x_points) - len(y))
        elif len(y) > len(x_points):
            y = y[:len(x_points)]

        h = plot_stair(ax, x_points, y, mstyle)

        # 收集统一图例句柄（以 label 为 key 去重）
        legend_handles[mstyle["label"]] = h
        legend_labels[mstyle["label"]] = mstyle["label"]

    # 坐标轴与标题
    ax.set_title(hardware)
    ax.set_xlabel("Iterations")
    ax.set_ylabel("HV")

    # 面板标注：放在坐标轴下方（相对坐标 transform=ax.transAxes）
    # x 位置可视美化，y 负值将文字放到坐标轴下方
    # ax.text(0.02, -0.22, f"{panel_tags[idx]} {hardware}",
    #         transform=ax.transAxes, ha="left", va="top")

# 统一图例：放到整张图上方
handles = list(legend_handles.values())
labels = list(legend_labels.values())

# 顶部留白以容纳图例
plt.subplots_adjust(top=0.83, wspace=0.25, hspace=0.35)

fig.legend(handles, labels,
           loc="upper center",
           ncol=min(5, len(labels)),
           frameon=False,
           bbox_to_anchor=(0.5, 0.97),
           handlelength=2.2,
           columnspacing=1.2)

# 统一 x 轴范围（可选）
for ax in axs.ravel():
    ax.set_xlim(x_points[0], x_points[-1])

# 保存与展示
out_path = f"{model}-hv-stair-2x2.pdf"
# os.makedirs(os.path.dirname(out_path), exist_ok=True)
plt.savefig(out_path, bbox_inches="tight")

plt.show()
