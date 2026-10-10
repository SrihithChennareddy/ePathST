"""Subcommand dispatch for the numbered multi-stage driver scripts (24, 25, 26).
Each stage keeps the command-line options it had as a standalone script."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path


def _call_main(module, entry: str):
    if entry == "run_dir":
        return module.main(sys.argv[1].rstrip("/"))
    if entry == "path":
        return module.main(Path(sys.argv[1]))
    return module.main()


def dispatch(prog: str, description: str, stages: dict[str, tuple[str, str, str]]) -> None:
    """Run `prog <stage> [stage options]`; `stages` maps name -> (module, entry style, help).

    Entry style "cli": the module parses `sys.argv` itself. "run_dir" and "path": the first
    positional argument is passed to `main` (as a string without trailing slash, or a Path).
    """
    if len(sys.argv) < 2 or sys.argv[1] in ("-h", "--help"):
        print(f"{description}\n\nusage: {prog} <stage> [stage options]\n\nstages:")
        for name, (_, _, help_text) in stages.items():
            print(f"  {name:<20} {help_text}")
        sys.exit(0 if len(sys.argv) >= 2 else 2)
    stage = sys.argv[1]
    if stage not in stages:
        sys.exit(f"unknown stage '{stage}'; choose from: {', '.join(stages)}")
    module_name, entry, _ = stages[stage]
    sys.argv = [f"{prog} {stage}", *sys.argv[2:]]
    _call_main(importlib.import_module(module_name), entry)
