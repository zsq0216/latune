# knobs_files

Parameter (“knobs”) specifications used by tuners and evaluators.

This folder contains:
- Per-hardware subfolders with **model-specific knob files** (used by `latune`).
- A **generic** `knobs_raw.json` (used by other methods unless stated otherwise).

> Convention in this repo:
> - `latune` reads: `knobs_files/<hardware>/<model>-<quant>.json`
> - Other methods read: `knobs_files/knobs_raw.json`

---

## Layout

knobs_files/
├── knobs_raw.json # generic knobs (fallback / non-latune methods)
├── rtx3060/
│ ├── phimoe-mini-q4.json # hardware+model specific knobs (latune)
│ └── qwen3-4b-q8.json
├── rtx4090/
│ └── ...
├── m4/
│ └── ...
└── orin/
└── ...


---

## File schema

```json
{
    "gpu-layers": {
        "default": 0,
        "type": "integer",
        "values": {
            "min": 0,
            "max": 100
        },
        "rank": 1
    },
    "no-kv-offload": {
        "default": false,
        "type": "boolean",
        "values": [
            true,
            false
        ],
        "rank": 2
    },
    ...
}