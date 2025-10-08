# -*- coding: utf-8 -*-
# NOTE: Execution code is commented out. We use hardcoded results to plot directly.

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # 必须在 import pyplot 之前设置 防止启动GUI
import matplotlib.pyplot as plt

# -----------------------------
# (A) Hardcoded results (from your printouts)
# -----------------------------
results = [
    {'tps_avg': np.float64(88.14437870898824), 'pps_avg': np.float64(522.3555617388661), 'cpu_avg': np.float64(30.9703125), 'mem_avg': np.float64(687.4296875), 'gpu_avg': np.float64(5024.5), 'config_name': 'default', 'gpu-layers': 32},
    {'tps_avg': np.float64(126.41361214190864), 'pps_avg': np.float64(567.558651548956), 'cpu_avg': np.float64(4.5046875), 'mem_avg': np.float64(586.24609375), 'gpu_avg': np.float64(5127.0), 'config_name': 'gpu-layers', 'gpu-layers': 100},
    # {'tps_avg': np.float64(88.48031115146114), 'pps_avg': np.float64(519.9184829115399), 'cpu_avg': np.float64(31.0484375), 'mem_avg': np.float64(696.68359375), 'gpu_avg': np.float64(5800.5), 'config_name': 'ctx-size', 'gpu-layers': 32, 'ctx-size': 8192},
    {'tps_avg': np.float64(74.44207496229502), 'pps_avg': np.float64(394.0985236728577), 'cpu_avg': np.float64(28.588750000000005), 'mem_avg': np.float64(1300.02109375), 'gpu_avg': np.float64(4293.6), 'config_name': 'no-kv-offload', 'gpu-layers': 32, 'no-kv-offload': True},
    {'tps_avg': np.float64(90.94985879679892), 'pps_avg': np.float64(514.1945860940518), 'cpu_avg': np.float64(28.673437500000002), 'mem_avg': np.float64(550.4921875), 'gpu_avg': np.float64(4921.0), 'config_name': 'flash-attn', 'gpu-layers': 32, 'flash-attn': True},
    {'tps_avg': np.float64(86.5668081551003), 'pps_avg': np.float64(515.2107085986222), 'cpu_avg': np.float64(25.17875), 'mem_avg': np.float64(690.3828125), 'gpu_avg': np.float64(5024.8), 'config_name': 'parallel', 'gpu-layers': 32, 'parallel': 8},
    {'tps_avg': np.float64(88.94260150045325), 'pps_avg': np.float64(516.6719898678776), 'cpu_avg': np.float64(30.771875), 'mem_avg': np.float64(687.91015625), 'gpu_avg': np.float64(5024.5), 'config_name': 'no-cont-batching', 'gpu-layers': 32, 'no-cont-batching': True},
    {'tps_avg': np.float64(88.88475090332543), 'pps_avg': np.float64(519.1062748913856), 'cpu_avg': np.float64(30.3703125), 'mem_avg': np.float64(735.73828125), 'gpu_avg': np.float64(5913.0), 'config_name': 'ubatch-size', 'gpu-layers': 32, 'ubatch-size': 4096},
]

df = pd.DataFrame(results)

required = {"config_name", "tps_avg", "gpu_avg"}
missing = required - set(df.columns)
if missing:
    raise ValueError(f"Missing required fields: {missing}")

base = df.loc[df["config_name"] == "default"].iloc[0]
tps_base = float(base["tps_avg"])
gpu_base = float(base["gpu_avg"])

df["tps_delta_pct"] = (df["tps_avg"].astype(float) - tps_base) / tps_base
df["gpu_delta_pct"] = (df["gpu_avg"].astype(float) - gpu_base) / gpu_base

# Remove default for plotting; sort so that LARGER changes are on TOP (上面大下面小)
plot_tps = (
    df.loc[df["config_name"] != "default", ["config_name", "tps_delta_pct"]]
      .set_index("config_name")
      .sort_values("tps_delta_pct", key=lambda s: s.abs(), ascending=True)
)
plot_gpu = (
    df.loc[df["config_name"] != "default", ["config_name", "gpu_delta_pct"]]
      .set_index("config_name")
      .sort_values("gpu_delta_pct", key=lambda s: s.abs(), ascending=True)
)

# -----------------------------
# (D) Plot tornado charts (two separate PDFs)
# -----------------------------
# 创建1行2列的子图布局
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 4.5), sharey=True, constrained_layout=True)


# 绘制TPS图 (左侧)
ax1.barh(plot_tps.index, plot_tps["tps_delta_pct"], color="#3083be")
# ax1.set_title("(a) TPS Relative Change (vs. default)", fontsize=22)
ax1.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x*100:.0f}%"))
ax1.axvline(0, linewidth=1, color='k')
ax1.grid(axis="x", linestyle="--", alpha=0.6)
ax1.tick_params(axis="both", labelsize=26)

# 绘制VRAM图 (右侧)
ax2.barh(plot_gpu.index, plot_gpu["gpu_delta_pct"], color="#d27d32")
# ax2.set_title("(b) VRAM Relative Change (vs. default)", fontsize=22)
ax2.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x*100:.0f}%"))
ax2.axvline(0, linewidth=1, color='k')
ax2.grid(axis="x", linestyle="--", alpha=0.6)
ax2.tick_params(axis="both", labelsize=26)

# 仅在左侧子图显示y轴标签（避免重复）
ax1.set_ylabel('')
ax2.set_ylabel('')  # 清空右侧y轴标签

# 设置x轴标签（仅在底部显示）
# 增加底部距离避免标签和x轴刻度重叠
ax1.set_xlabel('TPS Relative Change (%)', fontsize=26, labelpad=20)
ax2.set_xlabel('VRAM Relative Change (%)', fontsize=26, labelpad=20)

# 优化布局
plt.tight_layout(pad=3.0)  # 增加间距避免标签重叠
plt.savefig("tornado_combined.pdf", format="pdf", dpi=300, bbox_inches="tight")
plt.figure(figsize=(8, 2))
plt.close(fig)

# -----------------------------
# (E) Print a sorted summary table
# -----------------------------
def pct(x): return f"{x*100:.2f}%"
summary = (
    df[["config_name", "tps_avg", "gpu_avg", "tps_delta_pct", "gpu_delta_pct"]]
      .loc[df["config_name"] != "default"]
      .sort_values("tps_delta_pct", key=lambda s: s.abs(), ascending=False)
)
print("\n=== Sorted Summary by |TPS change| (largest to smallest) ===")
print(
    summary.assign(
        tps_delta_pct=summary["tps_delta_pct"].map(pct),
        gpu_delta_pct=summary["gpu_delta_pct"].map(pct),
    ).to_string(index=False)
)

print("\nSaved files:")
print(" - tornado_tps.pdf")
print(" - tornado_gpu.pdf")
