import json
import os

# 输入文件
input_files = ["low.json", "mid.json", "high.json"]

# 用于存放按 model 分类的数据
# model_data = {"phimoe-mini-q4","phimoe-mini-q8","qwen3-4b-q4", "qwen3-4b-q8"}
model_data = {}

# 逐个读取文件
for file in input_files:
    with open(file, "r", encoding="utf-8") as f:
        data = json.load(f)  # 假设每个文件里是一个列表
        for item in data:
            model = item["model"]
            if model not in model_data:
                model_data[model] = []
            model_data[model].append(item)

# 将每个 model 的数据写入单独文件
for model, items in model_data.items():
    filename = f"{model}.json"
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(items, f, indent=4, ensure_ascii=False)

print("处理完成，已生成各个 {model}.json 文件")
