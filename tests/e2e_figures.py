"""Regenerates the paper figures from figures/ and compares them pixel by pixel with the reference
images. Needs the HEST-1k benchmark data and slide scans (see figures/README.md); writes only to the
given folder.

    python tests/e2e_figures.py OUT_DIR [REFERENCE_ROOT]
"""

import os
import subprocess
import sys
from pathlib import Path

from PIL import Image, ImageChops

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REFERENCE = Path("/lustre/nvwulf/home/schennareddy/cellvit_project/paper_figures")

# script -> {generated file (relative to OUT_DIR): reference file (relative to the reference root)}
CHECKS = {
    "fig_compute_vs_pcc.py": {
        "tflops_vs_hest1k_pcc_ieee_trend_pareto.png": "final_paper_figures/tflops_vs_hest1k_pcc_ieee_trend_pareto.png",
    },
    "fig_leiden_tenx149.py": {
        "leiden_tenx149_comparison.png": "final_paper_figures/leiden_tenx149_comparison.png",
    },
    "fig_spatial_predictions.py": {
        f"ePathST_vs_HOptimus0_two_gene_panel{tag}.png": f"final_paper_figures/ePathST_vs_HOptimus0_two_gene_panel{tag}.png"
        for tag in ("", "_greens", "_ylgn", "_bugn", "_gnbu", "_summer")
    },
    "fig_gene_context.py": {"gene_context_dotplot.png": "gene_context_dotplot.png"},
    "fig_pretraining_data.py": {
        f"fig3_pretraining_data/{n}.png": f"final_paper_figures/fig3_pretraining_data/{n}.png"
        for n in (
            "fig3_combined",
            "fig3a_slides",
            "fig3b_tiles",
            "fig3c_access",
            "fig3d_size_score",
        )
    },
    "fig_data_efficiency.py": {
        "fig3_pretraining_data/fig3_data_efficiency.png": "final_paper_figures/fig3_pretraining_data/fig3_data_efficiency.png",
    },
}


def same_pixels(a: Path, b: Path) -> bool:
    with Image.open(a) as ia, Image.open(b) as ib:
        if ia.size != ib.size:
            return False
        return ImageChops.difference(ia.convert("RGBA"), ib.convert("RGBA")).getbbox() is None


def main():
    out_dir = Path(sys.argv[1]).resolve()
    reference = Path(sys.argv[2]) if len(sys.argv) > 2 else DEFAULT_REFERENCE
    out_dir.mkdir(parents=True, exist_ok=True)
    env = {**os.environ, "EPATHST_FIG_OUT": str(out_dir), "MPLBACKEND": "Agg"}
    failures = []
    for script, files in CHECKS.items():
        run = subprocess.run(
            [sys.executable, str(ROOT / "figures" / script)],
            env=env,
            cwd=ROOT / "figures",
            capture_output=True,
            text=True,
        )
        if run.returncode != 0:
            print(f"{script}: FAILED\n{run.stderr[-800:]}")
            failures.append(script)
            continue
        for generated, ref in files.items():
            ok = (out_dir / generated).exists() and same_pixels(
                out_dir / generated, reference / ref
            )
            print(f"{script}: {generated}: {'identical' if ok else 'DIFFERS'}")
            if not ok:
                failures.append(generated)
    print("PASS" if not failures else f"FAIL {failures}")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
