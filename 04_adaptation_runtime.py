import argparse
import json
import os
from config_evaluator import ConfigEvaluator

# Fixed list of tuning methods (NOT read from CLI)
METHODS = ["latune", "Default", "CBO", "SCOOT", "GA"]

# Optional reference of resource presets (kept for completeness; not used directly below)
RESOURCE_CONFIGS = [
    {"command": "python system_load_simulator.py", "resource": "low"},
    {"command": "python system_load_simulator.py --cpu 4 --memory 24576 --gpu-calc 1 --gpu-mem 4", "resource": "mid"},
    {"command": "python system_load_simulator.py --cpu 6 --memory 49152 --gpu-calc 2 --gpu-mem 6", "resource": "high"},
]


def parse_args():
    """
    Parse command-line arguments:
      --hardware: target device class (affects file paths)
      --model: base model family (will be combined with --quant)
      --quant: quantization level (combined with model name)
      --resource: system load (low/mid/high) 
    """
    parser = argparse.ArgumentParser(
        description="Evaluate tuning methods for a given hardware/model/quant configuration."
    )
    parser.add_argument(
        '--hardware',
        type=str,
        choices=['rtx3060', 'rtx4090', 'm4', 'orin'],
        default='rtx3060',
        help='Target hardware identifier (controls path layout and device assumptions).'
    )
    parser.add_argument(
        '--model',
        type=str,
        choices=['qwen3-4b', 'phimoe-mini'],
        default='qwen3-4b',
        help='Model family (will be combined with --quant, e.g., "phimoe-mini-q4").'
    )
    parser.add_argument(
        '--quant',
        type=str,
        choices=['q4', 'q8'],
        default='q4',
        help='Quantization level for the selected model.'
    )
    parser.add_argument(
        '--resource',
        type=str,
        choices=['low', 'mid', 'high'],
        default='low',
        help='System load preset.'
    )
    return parser.parse_args()


def main():
    """
    Main entry:
      - Build the full model name "<model>-<quant>".
      - For each method in METHODS:
          * Resolve parameter file and Pareto front path.
          * Run a single evaluation via ConfigEvaluator.
          * Aggregate results and write them to results/<hardware>/high.json.
    """
    args = parse_args()

    # Compose full model identifier as "<model>-<quant>", e.g., "phimoe-mini-q4"
    model_full = f"{args.model}-{args.quant}"

    hardware = args.hardware
    resource_rank = args.resource

    results = []

    print(f"=== Hardware: {hardware} | Model: {model_full} ===")

    for method in METHODS:
        print(f"Evaluating with method: {method}, model: {model_full}")

        # Path to the Pareto front file for this (hardware, model, method)
        pareto_front_path = f"pareto_fronts/{hardware}/{model_full}-{method}.json"

        # Skip this method if the Pareto file is missing
        if not os.path.exists(pareto_front_path):
            print(f"[SKIP] Pareto front not found: {pareto_front_path}")
            continue

        # Choose parameter file: method-specific for 'latune', otherwise a generic knobs file
        if method == 'latune':
            parameters_path = f"knobs_files/{hardware}/{model_full}.json"
        else:
            parameters_path = "knobs_files/knobs_raw.json"

        # Create evaluator
        config_evaluator = ConfigEvaluator(
            parameters_path=parameters_path,
            pareto_front_path=pareto_front_path,
            device="gpu"
        )

        # Run a single evaluation and capture its result (handle typical ValueError cases gracefully)
        try:
            result = config_evaluator.evaluate_instance(method=method, model=model_full)
            print(result)
        except ValueError as e:
            result = {"error": "ValueError", "message": str(e)}

        # Annotate result with metadata and append to the collection
        result["resource"] = resource_rank
        result["method"] = method
        result["model"] = model_full
        results.append(result)

    # Ensure output directories exist
    os.makedirs(f"results/{hardware}", exist_ok=True)

    # Write aggregated results
    output_path = f"results/{hardware}/{resource_rank}.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=4)

    print(f"All experiments completed. Results saved to {output_path}")


if __name__ == "__main__":
    main()
