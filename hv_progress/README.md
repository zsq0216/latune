# hv_progress

Hypervolume (HV) progress logs across iterations/budgets for multi-objective tuning.

## What’s here
- `.json` time series per run showing how Pareto HV evolves.

## File naming
- `./<hardware>/[-<model>-<quant>]-<method>.json` (e.g., `rtx3060/phimoe-mini-q4-latune.json`)

## Schema
```json
[
    0.6183876647228837,
    0.6716107985187011,
    0.6918313522594708,
    0.6957727924071307,
    0.6957727924071307,
    0.6978830493196598,
    0.6978830493196598,
    0.6979159189355878,
    0.69824855972296,
    0.69824855972296
]