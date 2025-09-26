#!/bin/bash

python 01_knobs_ranking.py --hardware rtx3060 --model qwen3-4b --quant q4
python 01_knobs_ranking.py --hardware rtx3060 --model qwen3-4b --quant q8


python 02_server_workflow.py --hardware rtx3060 --model qwen3-4b --quant q4
python 02_server_workflow.py --hardware rtx3060 --model qwen3-4b --quant q8


python 03_baseline_workflow.py --hardware rtx3060 --method Default --model qwen3-4b --quant q4
python 03_baseline_workflow.py --hardware rtx3060 --method Default --model qwen3-4b --quant q8
python 03_baseline_workflow.py --hardware rtx3060 --method GA --model qwen3-4b --quant q4
python 03_baseline_workflow.py --hardware rtx3060 --method GA --model qwen3-4b --quant q8
python 03_baseline_workflow.py --hardware rtx3060 --method CBO --model qwen3-4b --quant q4
python 03_baseline_workflow.py --hardware rtx3060 --method CBO --model qwen3-4b --quant q8


