# pareto_fronts

Saved Pareto-optimal configurations and their objective values for each method.

## What’s here
- `.json` files with the final Pareto set per (hardware, model, method).

## File naming
- `<hardware>/<model>-<method>.json` (e.g., `rtx3060/phimoe-mini-q4-latune.json`)

## Schema
```json
{
    "config": {
      "gpu-layers": 12,
      "no-kv-offload": true,
      "ctx-size": 3072,
      "ubatch-size": 1556,
      "draft": 10
    },
    "perf": {
      "tps_avg": 34.963530978907464,
      "gpu_avg": 1938.8
    }
}