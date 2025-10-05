import json
from pathlib import Path
from llama_executor import LlamaExecutor

def load_pareto_front(filepath: str):
    """从文件中加载 Pareto 前沿"""
    filepath = Path(filepath)
    if not filepath.exists():
        print(f"文件 {filepath} 不存在。")
        return

    with open(filepath, 'r', encoding='utf-8') as f:
        pareto_data = json.load(f)

    pareto_front = [
        item["config"] for item in pareto_data
    ]
    print(f"Pareto 前沿已从 {filepath} 加载。")
    return pareto_front

if __name__ == "__main__":
    param_types_instance ={'gpu-layers': 'integer',
                           'ctx-size': 'integer',
                           'no-kv-offload': 'boolean',
                           'draft': 'int',
                           'ubatch-size': 'integer',}
    config_list = load_pareto_front("insight/pareto_front.json")
    resource = "high"

    executor = LlamaExecutor(param_types=param_types_instance,
                             model_path="./../models/phimoe-mini-q4.gguf",
                              device="gpu")
    results = []
    for config in config_list:
        # print(config)
        result = executor.run_server_performance_test(config)
        # 把config拼接到result中
        # result.update(config)
        #把序号插入
        result.update({"number": f"{len(results)+1}"})
        results.append(result)
        print(result)
    
    # 对results按qps排序
    results = sorted(results, key=lambda x: x['tps_avg'], reverse=True)
    with open(f"insight/pareto_evaluated_{resource}.json", "w") as f:
        json.dump(results, f, indent=4)
