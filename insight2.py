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
        "A": {"gpu-layers":36},
        "B": {"gpu-layers": 32},
        "C": {"gpu-layers":36, "no-kv-offload": True},
        "D": {"gpu-layers":36, "flash-attn": True},
        "E": {"gpu-layers":36, "ubatch-size": 2048}
    }


    # 3) Executor initialization
    executor = LlamaExecutor(
        param_types=param_types_instance,
        model_path="./../models/phimoe-mini-q4.gguf",
        device="gpu",
    )

    # 4) Run all configurations
    results = []
    for name, cfg in config_options.items():
        result = executor.run_server_performance_test(cfg)
        result.update({"config_name": name})
        result.update(cfg)
        results.append(result)
        print(result)


