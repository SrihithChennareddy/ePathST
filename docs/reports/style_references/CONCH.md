# CONCH — public release quality

https://github.com/mahmoodlab/CONCH (Mahmood Lab, *Nature Medicine*)

## Verified structure

```
CONCH/
├── README.md, LICENSE, MANIFEST.in
├── pyproject.toml
├── conch/              # the installable package
│   ├── downstream/
│   └── open_clip_custom/
├── docs/
├── notebooks/
└── prompts/
```

## Verified `pyproject.toml`

```toml
[build-system]
requires = ["setuptools>=61.0"]
build-backend = "setuptools.build_meta"

[project]
name = "conch"
version = "0.1.0"
description = "CONCH: a vision language foundation model for human pathology."
readme = "README.md"
requires-python = ">=3.9"
dependencies = ["torch>=2.0.1", "torchvision", "transformers", ...]

[tool.setuptools.packages.find]
exclude = ["assets*", "benchmark*", "docs", "dist*", "scripts*", "tests*", "checkpoints*"]
```

## Verified install flow (README)

```shell
git clone https://github.com/mahmoodlab/CONCH.git
cd CONCH
conda create -n conch python=3.10 -y
conda activate conch
pip install -e .
```

## How this maps onto ePathST

ePathST currently has **no** `pyproject.toml` — all 26 scripts work around
this with `sys.path.insert(0, ...)`. This is explicitly Pass 5 (packaging)
scope, not a Pass 2-4 concern. When added:

- `name = "epathst"`, `src`-layout (`src/epathst/`), deps sourced from the
  existing `requirements.txt` (not re-derived).
- Exclude `scripts*`, `tests*`, `docs*`, `experimental_runs*`,
  `design*`, `slurm*` from the built package, same as CONCH excludes its own
  non-package directories.
- Scripts stay as plain scripts invoked with `python src/scripts/NN_*.py`,
  **not** converted to console-script entry points — CONCH's package exposes
  a model/library, not a CLI, and ePathST's numbered scripts are pipeline
  stages meant to be read and run in order, not general-purpose commands.
- README gets an "Installation" section modeled on this flow
  (`pip install -e .` once packaged) rather than the current implicit
  "add `src/` to your path" convention.
