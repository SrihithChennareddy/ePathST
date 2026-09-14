# PANTHER — scientific repository organization

https://github.com/mahmoodlab/PANTHER (Mahmood Lab)

## Verified structure

```
PANTHER/
├── README.md, LICENSE.md, env.yaml
├── docs/
└── src/
    ├── __init__.py
    ├── configs/
    ├── data_factory/
    ├── mil_models/
    ├── scripts/
    ├── splits/
    ├── training/
    ├── utils/
    ├── visualization/
    └── wsi_datasets/
```

A single `src/` package, subdivided by **conceptual responsibility** — data
loading, model definitions, splits, training loop, visualization, and
utilities each get their own top-level subpackage. `scripts/` lives inside
`src/` here (unlike ePathST, where `scripts/` sits next to `src/epathst/` —
kept that way because ePathST's scripts are numbered pipeline stages meant to
be run in sequence, not just entry points; see `Patho-Bench.md`).

## How this maps onto ePathST

The Pass 1 audit's proposed tree mirrors this split directly:

```
src/epathst/
├── data/          (cptac_select, wsi_patches, hest_prep, splits, expression, pathways, feature_cache)
├── models/        (teacher, student, patch_head, graph_layers, graph_model)
├── graph/         (builder, destruction, training)
├── distillation/  (training, evaluation)
├── evaluation/    (metrics, analysis, pathway_posthoc, genbio_comparison)
├── statistics/    (stats_tests)
├── efficiency/    (efficiency)
└── reporting/     (figures, report)
```

Deviation, justified by real dependencies rather than following PANTHER
mechanically: `ablations.py` and `utils.py` stay at the package root instead
of being filed under any one subpackage, because both are genuinely
cross-cutting (imported by nearly every subpackage) — PANTHER's own
`utils/` is the same kind of cross-cutting exception to its otherwise strict
split.
