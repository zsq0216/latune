import matplotlib.pyplot as plt
import numpy as np

# 数据（可按你的真实数值替换）
services = ["A", "B", "C", "D", "E"]
configs = ["Conf.A", "Conf.B", "Conf.C", "Conf.D", "Conf.E"]

# 0.0 用来表示“Error”（不画柱，改为文字标注）
# vals = np.array([
#     [126.26, 133.67, 126.73, 88.87, 100.87],  
#     [114.09, 131.10, 113.31, 82.05, 84.17],  
#     [90.04, 93.66, 90.78, 61.31, 76.33],  
#     [42.97, 45.52, 42.95, 35.55, 36.79],  
#     [45.38, 46.12, 45.26, 31.07, 40.88],  
# ])

vals = np.array([
    [1.   ,      0.876246,  0.991469 , 0 , 0],
 [1.057619 ,  1.   ,      1.032534 , 0 , 0],
 [1.005956 ,  0.864287  ,1.    ,     0 , 0],
 [0.701499 ,  0.627021 , 0.675292 , 1 , 0.775],
 [0.797665 ,  0.642455 , 0.840598 , 1.03 , 1]])

colors = ["#70de45", "#4bbfe5", "#d575dc", "#e6be5a", "#a85c3b"]
# hatches = ["////", "xxxx", "////", "", "...."]

x = np.arange(len(services))
width = 0.16
offsets = np.linspace(-2, 2, len(configs)) * width

fig, ax = plt.subplots(figsize=(3.8, 1.9), dpi=200)

# 画柱 + “Error” 标注
for i, (label, color) in enumerate(zip(configs, colors)):
    y = vals[i]
    bars = ax.bar(
        x + offsets[i], y, width=width, label=label,
        color=color, edgecolor="black", linewidth=0.6
    )
    for xi, yi in zip(x + offsets[i], y):
        if yi <= 1e-3:
            ax.text(xi, 0.05, "Error", rotation=90, ha="center",
                    va="bottom", fontsize=6, color="#d2691e")

# 坐标轴、网格、刻度
# ====== 修改纵轴范围 ======
# 自动设为 [0, 最大值 + 10%]
y_max = np.nanmax(vals)
ax.set_ylim(0, y_max)   # 自动适配范围

# 坐标轴、网格、刻度
ax.set_xlim(x[0] - 0.6, x[-1] + 0.6)
ax.set_ylabel("Relative TPS", fontsize=8)
ax.set_xlabel("Tuning Tasks", fontsize=8)
ax.set_xticks(x)
ax.set_xticklabels(services, fontsize=8)

# 动态设置 y 轴刻度
yticks = np.arange(0, 1.7, 0.5)
ax.set_yticks(yticks)
ax.set_yticklabels([f"{t:.1f}x" for t in yticks], fontsize=8)

# 网格和基线
ax.grid(axis="y", linestyle="--", linewidth=0.6, alpha=0.5)
ax.axhline(1.0, color="#E16D6D", linewidth=0.8)

# 图例
ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.20),
          ncol=5, fontsize=7, frameon=False, handlelength=1.5)

plt.tight_layout(pad=0.8)
plt.savefig("insight2.pdf", bbox_inches="tight")
plt.show()