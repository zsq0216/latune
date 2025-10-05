import json
import matplotlib.pyplot as plt

# 三个文件路径
files = {
    "low": "pareto_evaluated_low.json",
    "mid": "pareto_evaluated_mid.json",
    "high": "pareto_evaluated_high.json"
}

data = {}

# 读取每个文件
for label, path in files.items():
    with open(path, 'r', encoding='utf-8') as f:
        content = json.load(f)
        # 如果是单个对象，转成列表
        if isinstance(content, dict):
            content = [content]
        # 按 number 排序
        content.sort(key=lambda x: float(x["number"]))
        data[label] = {
            "x": [item["number"] for item in content],
            "y": [float(item["tps_avg"]) for item in content]
        }

# 画图
plt.figure(figsize=(12, 4))

for label, d in data.items():
    plt.plot(d["x"], d["y"], marker='o', label=label)
    # 标记 tps_avg == 0 的点为 ×
    for x, y in zip(d["x"], d["y"]):
        if y == 0:
            plt.scatter(x, y, marker='x', s=70, linewidths=2, color='red')

plt.xlabel("configuration")
plt.ylabel("TPS")
# plt.title("tps_avg 对比（low / mid / high）")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.5)
plt.tight_layout()
plt.savefig("insight3.pdf", dpi=150)
plt.show()
