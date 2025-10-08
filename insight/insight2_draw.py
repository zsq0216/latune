import matplotlib.pyplot as plt
import numpy as np

# 数据（可按你的真实数值替换）
services = ["A", "B", "C", "D", "E"]
configs = ["Conf.A", "Conf.B", "Conf.C", "Conf.D", "Conf.E"]

vals = np.array([
    [1.   ,      0.876246,  0.991469 , 0 , 0],
    [1.057619 ,  1.   ,      1.032534 , 0 , 0],
    [1.005956 ,  0.864287  ,1.    ,     0 , 0],
    [0.701499 ,  0.627021 , 0.675292 , 1 , 0.775],
    [0.797665 ,  0.642455 , 0.840598 , 1.03 , 1.0]
])

colors = ["#578e41", "#44a0be", "#aa7ead", "#ccac5d", "#a85c3b"]

x = np.arange(len(services))
width = 0.15
offsets = np.linspace(-2, 2, len(configs)) * width

# ---------- 更扁：更宽更矮 + 自动布局 ----------
fig, ax = plt.subplots(figsize=(6.4, 2.0), dpi=300, constrained_layout=True)

# 画柱 + “Error” 标注
for i, (label, color) in enumerate(zip(configs, colors)):
    y = vals[i]
    bars = ax.bar(
        x + offsets[i], y, width=width, label=label,
        color=color, edgecolor="black", linewidth=0.5
    )

# ---------- 纵轴：自适应最大值（更舒展） ----------
data_max = float(np.nanmax(vals))
# 顶部余量 15%，并设置一个“最低上限”避免图形显得矮；也避免超过 2.0 时过高
upper = max(1.3, data_max * 1.15)
ax.set_ylim(0, upper)

# 根据最终 ylim 放置 Error 文本（不贴底，不越界）
error_y = max(upper * 0.06, 0.06)
for i in range(len(configs)):
    for j in range(len(services)):
        if vals[i, j] <= 1e-3:
            ax.text(x[j] + offsets[i], error_y, "Error", rotation=90,
                    ha="center", va="bottom", fontsize=9, color="#d2691e")

# ---------- 字号 ----------
label_fs = 17
tick_fs = 13
legend_fs = 12

# x 轴范围按最外柱计算，留一点边距
ax.set_xlim((x[0] + offsets[0]) - (width*1.4), (x[-1] + offsets[-1]) + (width*1.4))

ax.set_ylabel("Relative TPS", fontsize=label_fs)
ax.set_xlabel("Tuning Tasks", fontsize=label_fs)
ax.set_xticks(x)
ax.set_xticklabels(services, fontsize=tick_fs)

# ---------- 动态 y 轴刻度 ----------
rng = upper
if rng <= 1.2:
    step = 0.2
elif rng <= 1.6:
    step = 0.25
else:
    step = 0.5
yticks = np.arange(0, upper + 1e-9, step)
# 确保 1.0 进入刻度
if not np.any(np.isclose(yticks, 1.0)):
    yticks = np.sort(np.unique(np.append(yticks, 1.0)))
ax.set_yticks(yticks)
ax.set_yticklabels([f"{t:.1f}x" for t in yticks], fontsize=tick_fs)

# 网格与基线
# ax.grid(axis="y", linestyle="--", linewidth=0.6, alpha=0.35)
ax.axhline(1.0, color="#E16D6D", linewidth=0.9)

# 轴线更素雅
for spine in ["top", "right"]:
    ax.spines[spine].set_visible(False)
ax.spines["left"].set_linewidth(0.8)
ax.spines["bottom"].set_linewidth(0.8)

# 图例移到下方，避免遮挡柱子
ax.legend(
    loc="upper center",
    bbox_to_anchor=(0.5, 1.25),  # 调整这个数字可微调高度
    ncol=5,
    fontsize=legend_fs,
    frameon=False,
    handlelength=1.3,
    columnspacing=0.8
)

plt.savefig("insight2.pdf", bbox_inches="tight")
plt.show()
