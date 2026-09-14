# timm — model implementation readability (PRIMARY reference)

https://github.com/huggingface/pytorch-image-models (Hugging Face / Ross Wightman)

## Verified structure

```
timm/
├── pyproject.toml, CITATION.cff
├── train.py, validate.py, inference.py, benchmark.py   # flat top-level scripts
└── timm/
    ├── data/, layers/, loss/, models/, optim/, scheduler/, task/, utils/
    └── version.py
```

Scripts (`train.py`, `validate.py`, ...) sit flat at the repo root, not
nested under the package — closer to ePathST's `src/scripts/` sitting beside
`src/epathst/` than to PANTHER/torchtune's `scripts/`-inside-package layout.

## Verified code sample (`timm/layers/mlp.py`, header + one class)

```python
""" MLP module w/ dropout and configurable activation layer

Hacked together by / Copyright 2020 Ross Wightman
"""
from functools import partial
from typing import Optional, Type, Union, Tuple

from torch import nn as nn


class Mlp(nn.Module):
    """ MLP as used in Vision Transformer, MLP-Mixer and related networks

    NOTE: When use_conv=True, expects 2D NCHW tensors, otherwise N*C expected.
    """
    def __init__(self, in_features, hidden_features=None, ...):
        ...

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.norm(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x
```

Observed conventions: module docstring is one line of substance (plus an
attribution line not relevant to ePathST); class docstring states what the
module implements and cites the source/paper it comes from; a `NOTE:` comment
flags a non-obvious shape/dtype contract (`use_conv=True` changes the
expected tensor layout) — exactly the "preserve comments describing units,
assumptions, non-obvious constraints" rule in `fix_code.txt` §7; `forward()`
itself has **zero** comments — each line is self-explanatory from the
variable names alone.

## How this maps onto ePathST

This is the direct model for how `graph_model.py`, `pna_block.py`,
`gatv2_block.py`, `student_model.py`, and `patch_head.py` were treated in the
comment-compression pass: forward-pass bodies are left uncommented where the
op sequence is self-explanatory, and multi-line docstrings were compressed to
one line **except** where they encode a real architectural contract — e.g.
`graph_model.ScaleAwareGraphCorrectionModel`'s class docstring is a formal
per-branch equation spec for the paper's A8 architecture and was deliberately
left multi-line, the same way timm would never compress a shape-contract
`NOTE:` into something lossy just to hit a line-count target.
