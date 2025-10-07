import json
import matplotlib.pyplot as plt

# ========== 全局大字体设置 ==========
plt.rcParams.update({
    "font.size": 22,         # 基础字体
    "axes.labelsize": 26,    # 坐标轴标签
    "xtick.labelsize": 22,
    "ytick.labelsize": 22,
    "legend.fontsize": 22,
})

# 四个文件路径
files = {
    "idle": "pareto_evaluated_idle.json",
    "low":  "pareto_evaluated_low.json",
    "mid":  "pareto_evaluated_mid.json",
    "high": "pareto_evaluated_high.json"
}

# 读取每个负载下的配置数据
raw_data = {}
for label, path in files.items():
    with open(path, 'r', encoding='utf-8') as f:
        content = json.load(f)
        if isinstance(content, dict):
            content = [content]
        content.sort(key=lambda x: float(x["number"]))
        raw_data[label] = {str(item["number"]): float(item["tps_avg"]) for item in content}

# 提取所有配置号（字符串形式）
config_numbers = sorted({num for dataset in raw_data.values() for num in dataset.keys()}, key=float)

# 负载顺序
load_levels = ["idle", "low", "mid", "high"]

# 为每个配置分配不同的 marker 和线型（不靠颜色区分）
markers = ['o', 's', '^', 'D', 'v', 'P', '*', 'X', '>', '<', 'h', 'H']
linestyles = ['-', '--', '-.', ':']

fig = plt.figure(figsize=(12, 6))

for idx, config in enumerate(config_numbers):
    y_values = [raw_data.get(load, {}).get(config, 0) for load in load_levels]

    marker = markers[idx % len(markers)]
    linestyle = linestyles[idx % len(linestyles)]
    line_color = 'black'  # 统一使用黑色，不用颜色区分

    # 将连续非零段分别绘制，避免穿过 0 点
    segment_x, segment_y = [], []
    added_label = False  # 只在第一次绘制该配置的线段时加图例标签

    for load, y in zip(load_levels, y_values):
        if y != 0:
            segment_x.append(load)
            segment_y.append(y)
        else:
            # 画出前一段（如果存在）
            if segment_x:
                plt.plot(
                    segment_x, segment_y,
                    marker=marker,
                    linestyle=linestyle,
                    color=line_color,
                    linewidth=2.5,
                    markersize=10,
                    label=(f'config {config}' if not added_label else None),
                )
                added_label = True
                segment_x, segment_y = [], []
            # 标记 0 的点为红色 ×
            plt.scatter(load, y, marker='x', s=180, linewidths=4.5, color='red', zorder=3)

    # 收尾：最后一段
    if segment_x:
        plt.plot(
            segment_x, segment_y,
            marker=marker,
            linestyle=linestyle,
            color=line_color,
            linewidth=2.5,
            markersize=10,
            label=(f'config {config}' if not added_label else None),
        )

plt.xlabel("Load Level")
plt.ylabel("TPS")
plt.grid(True, linestyle="--", alpha=0.5)

# 图例放在图内右上角
plt.legend(title="Configuration", loc='upper right', frameon=True, framealpha=0.9)

plt.tight_layout()
plt.savefig("insight3_by_config_noconnect0_bigfonts.pdf", dpi=300)
plt.show()
