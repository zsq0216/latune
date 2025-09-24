#!/bin/bash

# python 02_server_workflow.py --model phimoe-mini --quant q8
# python 02_server_workflow.py --model qwen3-4b --quant q4
# python 02_server_workflow.py --model qwen3-4b --quant q8

# python 03_baseline_workflow.py --method Default --model qwen3-4b --quant q4
# python 03_baseline_workflow.py --method Default --model qwen3-4b --quant q8
# python 03_baseline_workflow.py --method GA --model qwen3-4b --quant q4
# python 03_baseline_workflow.py --method GA --model qwen3-4b --quant q8
# python 03_baseline_workflow.py --method CBO --model qwen3-4b --quant q4
# python 03_baseline_workflow.py --method CBO --model qwen3-4b --quant q8

python 03_baseline_workflow.py --method Default --model phimoe-mini --quant q8
python 03_baseline_workflow.py --method GA --model phimoe-mini --quant q8
python 03_baseline_workflow.py --method CBO --model phimoe-mini --quant q8
