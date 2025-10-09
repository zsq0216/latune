from config_evaluator import ConfigEvaluator
import json

# Three resource workload configurations (commands to start workload simulator)
resource_configs = [
    {"command": "python workload_simulator.py", "resource": "low"},
    {"command": "python workload_simulator.py --cpu 4 --memory 24576 --gpu-calc 1 --gpu-mem 4", "resource": "mid"},
    {"command": "python workload_simulator.py --cpu 6 --memory 49152 --gpu-calc 2 --gpu-mem 6", "resource": "high"},
]

# Tuning methods to evaluate (you can add "latune", "Default", "CBO", "scoot" etc.)
methods = ["latune", "Default", "CBO", "scoot", "GA"]

# Container for results
results = []

hardware = "rtx3060"
resource_rank = "high"

# Models to evaluate (adjust list as needed)
for model in ["phimoe-mini-q4"]:
    for method in methods:
        print(f"Evaluating with method: {method}, model: {model}")

        # Pareto front file path for this hardware/model/method
        pareto_front_path = f"pareto_fronts/{hardware}/{model}-{method}.json"

        # Choose parameters file: method-specific or generic
        if method == 'latune':
            parameters_path = f"knobs_files/{hardware}/{model}.json"
        else:
            parameters_path = "knobs_files/knobs_raw.json"

        # Create evaluator instance
        config_evaluator = ConfigEvaluator(
            parameters_path=parameters_path,
            pareto_front_path=pareto_front_path,
            device="gpu"
        )

        try:
            # Run a single evaluation instance and collect the result
            result = config_evaluator.evaluate_instance(method=method, model=model)
            print(result)
        except ValueError as e:
            # Handle missing/invalid Pareto or budget cases
            result = {"error": "ValueError", "message": str(e)}

        # Annotate result with metadata
        result["resource"] = resource_rank
        result["method"] = method
        result["model"] = model
        results.append(result)

# Ensure output directory exists before writing if needed (not created here)
output_path = f"results/{hardware}/{resource_rank}-1.json"
with open(output_path, "w") as f:
    json.dump(results, f, indent=4)

print(f"All experiments completed. Results saved to {output_path}")
