import json
from pathlib import Path
from llama_executor import LlamaExecutor

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":
    # 1) Parameter type definitions
    param_types_instance = {
        'gpu-layers': 'integer',
        'ctx-size': 'integer',
        'no-kv-offload': 'boolean',
        'flash-attn': 'boolean',
        'parallel': 'integer',
        "no-cont-batching": "boolean",
        "thread": "integer",
        "ubatch-size": "integer"
    }

    # 2) Configuration list
    config_options = {
        "A": {"gpu-layers":3},
        "B": {"gpu-layers": 3,"flash-attn": True},
    }

    for model in ["phimoe-mini-q8"]:#"qwen3-4b-q4", "qwen3-4b-q8", "phimoe-mini-q4", 
    # 3) Executor initialization
        executor = LlamaExecutor(
            param_types=param_types_instance,
            model_path=f"./../models/{model}.gguf",
            device="gpu",
        )
        print (f"Evaluating model: {model}")
        # 4) Run all configurations
        results = []
        for name, cfg in config_options.items():
            result = executor.run_server_performance_test(cfg)
            result.update({"config_name": name})
            result.update(cfg)
            results.append(result)
            print(result)


