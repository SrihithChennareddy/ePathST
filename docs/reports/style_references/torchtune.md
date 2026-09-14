# torchtune — Python/PyTorch engineering style (PRIMARY reference)

https://github.com/meta-pytorch/torchtune (Meta)

## Verified structure

```
torchtune/
├── pyproject.toml, CONTRIBUTING.md, CITATION.cff
├── docs/
├── recipes/       # training recipes (the "scripts" layer)
├── tests/
└── torchtune/     # the library
    ├── config/, data/, datasets/, generation/
    ├── models/, modules/, rlhf/, training/, utils/
    └── _cli/
```

Same library-vs-orchestration split as PANTHER/Patho-Bench: `torchtune/` is
the reusable package, `recipes/` is where you actually run something.

## Verified code sample (`torchtune/modules/rms_norm.py`, in full)

```python
import torch
from torch import nn


class RMSNorm(nn.Module):
    """
    Root Mean Square Normalization in fp32.

    See: https://pytorch.org/docs/stable/generated/torch.nn.RMSNorm.html

    Args:
        dim (int): embedding size
        eps (float): small value to avoid division by zero. Default: 1e-6
    """

    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.normalized_shape = (dim,)
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): input tensor to normalize

        Returns:
            torch.Tensor: The normalized and scaled tensor having the same shape as ``x``.
        """
        # computation is in fp32
        x_fp32 = x.float()
        x_normed = (
            x_fp32 * torch.rsqrt(x_fp32.pow(2).mean(-1, keepdim=True) + self.eps)
        ).type_as(x)
        return x_normed * self.scale
```

Observed conventions: full type hints on every signature; one-line WHY
comments only where behavior isn't obvious from the code (`# computation is
in fp32`); Args/Returns docstrings kept short — one line per argument, no
restated prose; class docstring states purpose + a doc link, not an essay.

## How this maps onto ePathST

This is exactly the standard Pass 2/3 was applying by hand across the
codebase: module docstrings capped at ~2 lines, function docstrings
compressed to keep only non-obvious WHY content (units, invariants, bug
history) and drop restated WHAT, one-line comments over multi-line narration.
The one deliberate divergence: torchtune's Args/Returns block style wasn't
imposed wholesale on ePathST, since most ePathST functions already had
adequate inline type hints and forcing a full Args/Returns block on every
function would add exactly the kind of "giant docstring for a simple
function" `fix_code.txt` §7 says to avoid.
