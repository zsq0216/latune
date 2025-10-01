from config_evaluator import ConfigEvaluator
import json

# 三种资源状态配置
resource_configs = [
    {"command": "python workload_simulator.py", "resource": "low"},
    {"command": "python workload_simulator.py --cpu 4 --memory 24576 --gpu-mem 4", "resource": "mid"},
    {"command": "python workload_simulator.py --cpu 6 --memory 49152 --gpu-mem 5", "resource": "high"},
]

# 四种评估方法
methods = ["Default", "GA", "CBO", "scoot", "latune"]
# methods = ["CBO"]

# 结果列表
results = []

hardware = "rtx3060"
resource_rank ="high"

for model in ["qwen3-4b-q4","qwen3-4b-q8", "phimoe-mini-q4", "phimoe-mini-q8"]:
    for method in methods:
        print(f"Evaluating with method: {method}, model: {model}")
        pareto_front_path = f"pareto_fronts/{hardware}/{model}-{method}.json"
        if method == 'latune':
            parameters_path = f"knobs_files/{hardware}/{model}.json"
        else:
            parameters_path = f"knobs_files/knobs_raw.json"
        config_evaluator = ConfigEvaluator(
            parameters_path = parameters_path,
            pareto_front_path=pareto_front_path,
            device="gpu")
        try:
            result = config_evaluator.evaluate_instance(method=method, model=model)
        except ValueError as e:
            result = {"error": "ValueError"}
        result["resource"] = resource_rank
        result["method"] = method
        result["model"] = model
        results.append(result)

# 写入 JSON 文件
with open(f"results/{hardware}/{resource_rank}.json", "w") as f:
    json.dump(results, f, indent=4)

print("All experiments completed. Results saved to performance_results.json")
