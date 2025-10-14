# bounds

Reference bounds for metrics used by tuners and validators.

## What’s here
- `.json` files define metric priors for each model and hardware target.

## File naming
- `./<hardware>/<model>.json`

## Schema
```json
{
    "tps_avg": {
        "min": 0.0,
        "max": 60.12843291354783
    },
    "gpu_p95": {
        "min": 440.77777777777777,
        "max": 2443.1666666666665
    }
}
```