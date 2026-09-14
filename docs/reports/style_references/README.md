# Style references for the ePathST refactor

`fix_code.txt` (the governing spec for the presentation-only refactor on
`refactor/publication-quality`) names five repos as **style references only**
— never as a source of algorithms, defaults, or scientific logic to copy.
Each file in this folder records what was actually verified in that repo
(directory structure, a representative source file, packaging config — pulled
live via the GitHub API, not recalled from memory) and how it maps onto a
concrete decision in this refactor.

| Repo | Use it for | Notes |
|---|---|---|
| [PANTHER.md](PANTHER.md) | scientific repo organization | models/data/training/splits/configs separation |
| [CONCH.md](CONCH.md) | public release quality | `pyproject.toml`, install flow, README shape |
| [Patho-Bench.md](Patho-Bench.md) | experiment/evaluation organization | configs, experiments, benchmarking layout |
| [torchtune.md](torchtune.md) | Python/PyTorch engineering style (primary) | docstrings, type hints, formatting |
| [timm.md](timm.md) | model implementation readability (primary) | concise `nn.Module`s, forward-pass clarity |

If you're extending ePathST later and want a second opinion on where new code
should live, or how to write/document it, check the matching file here before
inventing a new convention.
