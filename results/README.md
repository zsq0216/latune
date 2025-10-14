# results

Per-run runtime results aggregated across methods under specified budgets.

## What’s here
- `.json` files listing outcomes returned by `ConfigEvaluator` for a hardware target.

## File naming
- `<hardware>/<budget_or_tag>.json` (e.g., `rtx3060/high.json`)

## Schema
```json
[
    {
        "tps_avg": 44.35398137255015,
        "gpu_avg": 3350.0,
        "resource": "low",
        "method": "Default",
        "model": "qwen3-4b-q4"
    },
    {
        "tps_avg": 89.11730590996272,
        "gpu_avg": 2816.0,
        "resource": "low",
        "method": "GA",
        "model": "qwen3-4b-q4"
    },
]