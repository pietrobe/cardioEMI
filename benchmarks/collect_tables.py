"""Build the paper's assembly and linear solver tables from the benchmark results (standard library only).

    python3 benchmarks/collect_tables.py [results_dir]      # default: benchmarks/results

Expects <results_dir>/<model>_n<n>/<config>.log (stdout of main.py) and
<results_dir>/<model>_n<n>/<config>/stats-*.json (save_performance: True), as
written by benchmarks/run.sbatch. Prints Markdown tables and writes
<results_dir>/tables.tex and <results_dir>/summary.csv:

  Table 4  assembly strong scaling, A1 and B1: right-hand side assembly, seconds per time step
  Table 5  assembly weak scaling, A1-A4 and B1-B4: assembly MDoFs/s per node
  Table 6  solver strong scaling, A1: solve seconds per time step (iterations)
  Table 7  solver weak scaling, A1-A4: solve MDoFs/s per node (iterations)
  Table 8  solver strong scaling, B1 and B2: AMG-CG solve, seconds per time step (iterations)
  Table 9  solver weak scaling, B1-B4: solve MDoFs/s per node (iterations)

Definitions:
  - assembly s/step = (vector assembly time - its one-off setup in the first step) / time steps,
    from the AMG-CG runs;
  - solve s/step = KSP solve time / time steps (includes the AMG setup of the first solve);
  - MDoFs/s per node = DoFs / (s/step) / nodes / 1e6;
  - iterations = average number of Krylov iterations per time step.
"Nodes" are the paper's nodes: n GPUs (one rank each) for GPU runs, 144 n cores for CPU runs.
Missing runs are shown as "n/a", failed ones as "fail" (e.g. GPU out of memory).
"""

import argparse
import csv
import json
import re
import statistics
from pathlib import Path

NODES = (1, 2, 4, 8)
WEAK_A = [("A1", 1), ("A2", 2), ("A3", 4), ("A4", 8)]
WEAK_B = [("B1", 1), ("B2", 2), ("B3", 4), ("B4", 8)]
SOLVERS = ["gpu_cg_amg", "gpu_gmres_amg", "cpu_cg_amg", "cpu_gmres_amg"]
SOLVERS_A = ["gpu_cg_jacobi", "gpu_cg_amg", "cpu_cg_amg"]
HEAD = {"gpu_cg_amg": "GPU AMG-CG", "gpu_gmres_amg": "GPU AMG-GMRES",
        "cpu_cg_amg": "CPU AMG-CG", "cpu_gmres_amg": "CPU AMG-GMRES", "gpu_cg_jacobi": "GPU Jacobi-CG"}
REASON = re.compile(r"Linear solve (converged|did not converge) due to (\w+) iterations (\d+)")


def parse(res, model, n, config):
    """Result of one run, or None if its log is missing."""
    run_dir = res / f"{model}_n{n}"
    log = run_dir / f"{config}.log"
    if not log.exists():
        return None
    its, failed = [], False
    for line in log.read_text(errors="replace").splitlines():
        m = REASON.search(line)
        if m:
            its.append(int(m.group(3)))
            failed |= m.group(1) != "converged"
    r = {"steps": len(its), "avg_its": statistics.mean(its) if its else None, "failed": failed or not its,
         "dofs": None, "solve_step": None, "assembly_step": None}
    stats = sorted((run_dir / config).glob("stats-*.json"))
    if stats:
        doc = json.loads(stats[-1].read_text())
        perf, steps = doc["performance"], doc["input"]["time_steps"]
        r["failed"] |= bool(perf.get("failed")) or len(its) != steps
        r["dofs"] = perf.get("num_dofs")
        r["solve_step"] = perf["solve"] / steps
        r["assembly_step"] = (perf["vector_assemble"] - perf["vector_assemble_setup"]) / steps
    else:
        r["failed"] = True   # the run did not reach the end
    return r


def fmt(r, key, nodes=None, its=False):
    """Seconds per step, or MDoFs/s per node when nodes is given; iterations in parentheses."""
    if r is None:
        return "n/a"
    if r["failed"] or r[key] is None:
        return "fail"
    text = f"{r[key]:.2f}" if nodes is None else f"{r['dofs'] / r[key] / nodes / 1e6:.2f}"
    return f"{text} ({r['avg_its']:.0f})" if its else text


def markdown(title, first, cols, rows):
    out = [f"\n### {title}\n", "| " + " | ".join([first] + cols) + " |", "|" + "---|" * (len(cols) + 1)]
    return "\n".join(out + ["| " + " | ".join(row) + " |" for row in rows])


def latex(first, cols, rows):
    body = "\n".join(" & ".join(r) + " \\\\" for r in rows)
    return ("\\begin{tabular}{l" + "r" * len(cols) + "}\n\\toprule\n" + " & ".join([first] + cols)
            + " \\\\\n\\midrule\n" + body + "\n\\bottomrule\n\\end{tabular}\n")


def main(res):
    get = lambda m, n, c: parse(res, m, n, c)  # noqa: E731
    tables = []

    cols = ["A1 GPU", "A1 CPU", "B1 GPU", "B1 CPU"]
    rows = [[str(n)] + [fmt(get(m, n, f"{f}_cg_amg"), "assembly_step") for m in ("A1", "B1") for f in ("gpu", "cpu")]
            for n in NODES]
    tables.append(("Table 4: assembly strong scaling, seconds per time step", "Nodes", cols, rows))

    cols = ["A GPU", "A CPU", "B GPU", "B CPU"]
    rows = [[str(n)] + [fmt(get(m, n, f"{f}_cg_amg"), "assembly_step", n) for m, _ in (a, b) for f in ("gpu", "cpu")]
            for a, b, n in zip(WEAK_A, WEAK_B, NODES)]
    tables.append(("Table 5: assembly weak scaling, MDoFs/s per node (A1-A4, B1-B4)", "Nodes", cols, rows))

    cols = [HEAD[c] for c in SOLVERS_A]
    rows = [[str(n)] + [fmt(get("A1", n, c), "solve_step", its=True) for c in SOLVERS_A] for n in NODES]
    tables.append(("Table 6: Model A strong scaling (A1), solve seconds per time step (iterations)", "Nodes", cols, rows))

    rows = [[str(n)] + [fmt(get(m, n, c), "solve_step", n, its=True) for c in SOLVERS_A] for m, n in WEAK_A]
    tables.append(("Table 7: Model A weak scaling (A1-A4), solve MDoFs/s per node (iterations)", "Nodes", cols, rows))

    cols = ["B1 GPU", "B1 CPU", "B2 GPU", "B2 CPU"]
    rows = [[str(n)] + [fmt(get(m, n, f"{f}_cg_amg"), "solve_step", its=True) for m in ("B1", "B2") for f in ("gpu", "cpu")]
            for n in NODES]
    tables.append(("Table 8: AMG-CG strong scaling, solve seconds per time step (iterations)", "Nodes", cols, rows))

    cols = [HEAD[c] for c in SOLVERS]
    rows = [[m] + [fmt(get(m, n, c), "solve_step", n, its=True) for c in SOLVERS] for m, n in WEAK_B]
    tables.append(("Table 9: solver weak scaling, solve MDoFs/s per node (iterations)", "Model", cols, rows))

    for title, first, cols, rows in tables:
        print(markdown(title, first, cols, rows))
    tex = res / "tables.tex"
    tex.write_text("% Generated by benchmarks/collect_tables.py\n"
                   + "".join(f"% {title}\n" + latex(first, cols, rows) for title, first, cols, rows in tables))

    summary = res / "summary.csv"
    with open(summary, "w", newline="") as fp:
        w = csv.writer(fp)
        w.writerow(["run", "config", "steps", "avg_its", "failed", "dofs", "assembly_s_per_step", "solve_s_per_step"])
        for log in sorted(res.glob("*_n*/*.log")):
            model, n = log.parent.name.rsplit("_n", 1)
            r = parse(res, model, n, log.stem)
            w.writerow([log.parent.name, log.stem, r["steps"], r["avg_its"], r["failed"], r["dofs"],
                        r["assembly_step"], r["solve_step"]])
    print(f"\nwritten {tex} and {summary}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Build Tables 4-9 from the benchmark results.")
    ap.add_argument("results", nargs="?", default=str(Path(__file__).resolve().parent / "results"))
    main(Path(ap.parse_args().results))
