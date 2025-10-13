import argparse

from tuningworkflow import TuningWorkflow

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Llama Configuration Optimizer (Server Workflow)")
    parser.add_argument("--device", type=str, choices=["cpu", "gpu"], default="gpu")
    parser.add_argument("--hardware", type=str, choices=["rtx3060", "rtx4090", "m4", "orin"], default="rtx3060")
    parser.add_argument("--model", type=str, choices=["qwen3-4b", "phimoe-mini"], default="qwen3-4b")
    parser.add_argument("--quant", type=str, choices=["q4", "q8"], default="q4")
    parser.add_argument("--history_hardware", type=str, default="m4")
    args = parser.parse_args()

    parameters_path = f"knobs_files/{args.hardware}/{args.model}-{args.quant}.json"

    if args.device == "gpu":
        objectives = {"tps_avg": "max", "gpu_avg": "min"}
    else:
        objectives = {"tps_avg": "max", "mem_avg": "min"}

    model_name = f"{args.model}-{args.quant}"
    history_hw = args.history_hardware or args.hardware  # 原脚本里硬编码 m4，可用 --history_hardware=m4 复现
    history_path = f"pareto_fronts/{history_hw}/{model_name}-latune.json"

    print("======= START =======")
    print(f"model: {args.model}, quant: {args.quant}, hardware: {args.hardware}")
    print(f"Warm-start history: {history_path}")

    workflow = TuningWorkflow(
        parameters_path=parameters_path,
        objectives=objectives,
        max_observations=50,
        parallel_degree=5,
        device=args.device,
        hardware=args.hardware,
        model=args.model,
        quant=args.quant
    )

    workflow.run_workflow()

    print("\n=== Tuning Results ===")
    print(f"Total evaluations: {len(workflow.history['configs'])}")
    print("Best configuration:")
    # print(workflow.get_best_config())

    # 需要时再开启：
    workflow.save_pareto_front_and_hv(model_name)
