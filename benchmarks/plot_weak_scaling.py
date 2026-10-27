"""Draw the weak-scaling figures of the paper from the benchmark results (needs matplotlib).

    python3 benchmarks/plot_weak_scaling.py [results_dir]      # default: benchmarks/results

Writes <results_dir>/weak_scaling_structured.pdf (Model A, A1-A4 on 1-8 nodes) and
<results_dir>/weak_scaling_unstructured.pdf (Model B, B1-B4 on 1-8 nodes). Each figure has two
panels, average time per time step against the number of DoFs, on log-log axes:
  - Solver: linear solve (the curves of Table 7 or Table 9);
  - Assembly: right-hand side assembly from the AMG-CG runs (Table 5).
Runs are read as in collect_tables.py; failed or missing runs are left out of their curve.
"""

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from collect_tables import HEAD, WEAK_A, WEAK_B, parse  # noqa: E402

BLUE, LIGHT_BLUE, GREEN, ORANGE = "#0072bd", "#2e8cc9", "#77ac30", "#d95319"
STYLE = {
    "gpu_cg_amg": dict(color=BLUE, ls="--", marker="o"),
    "gpu_gmres_amg": dict(color=LIGHT_BLUE, ls=":", marker="s"),
    "gpu_cg_jacobi": dict(color=GREEN, ls="-.", marker="d"),
    "cpu_cg_amg": dict(color=ORANGE, ls="--", marker="o"),
    "cpu_gmres_amg": dict(color=ORANGE, ls="-", marker="^"),
}
FIGURES = {
    "weak_scaling_structured": ("Weak Scaling - Structured Mesh", WEAK_A,
                                ["gpu_cg_jacobi", "gpu_cg_amg", "cpu_cg_amg"]),
    "weak_scaling_unstructured": ("Weak Scaling - Unstructured Mesh", WEAK_B,
                                  ["gpu_cg_amg", "gpu_gmres_amg", "cpu_cg_amg", "cpu_gmres_amg"]),
}


def series(res, runs, config, key):
    pts = []
    for model, n in runs:
        r = parse(res, model, n, config)
        if r is not None and not r["failed"] and r["dofs"] and r[key]:
            pts.append((r["dofs"], r[key]))
    return pts


def panel(ax, title, curves):
    for label, config, pts in curves:
        if pts:
            x, y = zip(*pts)
            ax.plot(x, y, label=label, mfc="white", mew=1.5, lw=1.5, **STYLE[config])
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("DoFs")
    ax.set_title(title)
    ax.grid(True, which="both", alpha=0.3, lw=0.5)
    if any(pts for _, _, pts in curves):
        ax.legend(fontsize="small")


def main(res):
    for name, (title, runs, solvers) in FIGURES.items():
        solver = [(HEAD[c], c, series(res, runs, c, "solve_step")) for c in solvers]
        assembly = [(f, f"{f.lower()}_cg_amg", series(res, runs, f"{f.lower()}_cg_amg", "assembly_step"))
                    for f in ("GPU", "CPU")]
        if not any(pts for _, _, pts in solver + assembly):
            print(f"{name}: no completed runs in {res}, skipped")
            continue
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8.5, 3.8))
        panel(ax1, "Solver", solver)
        ax1.set_ylabel("Runtime per time step [s]")
        panel(ax2, "Assembly", assembly)
        fig.suptitle(title)
        fig.tight_layout()
        out = res / f"{name}.pdf"
        fig.savefig(out)
        plt.close(fig)
        print(f"written {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Draw the weak-scaling figures from the benchmark results.")
    ap.add_argument("results", nargs="?", default=str(Path(__file__).resolve().parent / "results"))
    main(Path(ap.parse_args().results))
