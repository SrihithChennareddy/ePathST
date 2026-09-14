# Patho-Bench — experiment/evaluation organization

https://github.com/mahmoodlab/Patho-Bench (Mahmood Lab)

## Verified structure

```
Patho-Bench/
├── README.md, LICENSE
├── pyproject.toml, requirements.txt
├── advanced_usage/
├── tutorial/
└── patho_bench/
    ├── DatasetFactory.py
    ├── ExperimentFactory.py
    ├── Pooler.py
    ├── Runner.py
    ├── SplitFactory.py
    ├── TrainableSlideEncoder.py
    ├── config/
    ├── datasets/
    ├── experiments/
    ├── helpers/
    ├── optim/
    └── scripts/
```

A `*Factory.py` per concern (dataset, experiment, split) plus a `Runner.py`
that orchestrates them — experiment *definitions* (config/experiments/) are
kept separate from experiment *execution* (Runner.py, scripts/).

## How this maps onto ePathST

ePathST already follows the same spirit without needing factories: the
numbered `src/scripts/01_...py` … `23_...py` pipeline **is** the "experiment
definition + execution" layer (each script = one reproducible experiment
stage), and `src/epathst/` is the reusable implementation each script calls
into — matching Patho-Bench's `patho_bench/` (library) vs `scripts/`
(orchestration) split.

`fix_code.txt` explicitly says not to introduce factories/registries here —
ePathST's ablation grid (`ablations.ABLATION_GRID`, a plain list of dicts) and
numbered-script dispatch already serve the role Patho-Bench's
`ExperimentFactory`/`DatasetFactory` classes serve, at a scale where a class
hierarchy would be overengineering, not clarity (`fix_code.txt` §13).
Patho-Bench's benefit taken here is narrower and more useful: config and
experiment logic living in clearly separate, discoverable files (already true
in ePathST via `configs/base.yaml` + the numbered scripts), not the
factory-class pattern itself.
