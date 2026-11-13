#!/usr/bin/env python3
"""Paired comparison of segmentation methods on the same held-out brains.

Reads the per-case score table written by ``score_hammers_stage_a.py`` (one row
per case and method) and prints only aggregate statistics: mean paired
difference, percentile-bootstrap 95% CI, Wilcoxon signed-rank p-value
(Holm-corrected across all comparisons), and the number of cases where the
reference method is better. No per-case values are written.

Usage:
  python code/evaluation/paired_stats.py hammers_stage_a.tsv --ref nnunet \
      --against maper jlf --out stats.json
  python code/evaluation/paired_stats.py --self-test
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon

# metric -> True if higher is better
METRICS = {
    "dice_L": True, "dice_R": True,
    "jaccard_L": True, "jaccard_R": True,
    "asd_L_mm": False, "asd_R_mm": False,
    "absdvol_L_mm3": False, "absdvol_R_mm3": False,
}


def load(path: Path) -> dict[str, dict[str, dict[str, float]]]:
    """method -> case -> metric -> value."""
    out: dict[str, dict[str, dict[str, float]]] = {}
    with open(path, newline="") as f:
        sample = f.readline()
        f.seek(0)
        reader = csv.DictReader(f, delimiter="\t" if "\t" in sample else ",")
        for row in reader:
            vals = {}
            for m in METRICS:
                if m.startswith("absdvol"):
                    v = row.get(m.replace("absdvol", "dvol"), "")
                    vals[m] = abs(float(v)) if v not in ("", "nan", "None") else np.nan
                else:
                    v = row.get(m, "")
                    vals[m] = float(v) if v not in ("", "nan", "None") else np.nan
            out.setdefault(row["method"], {})[row["case"]] = vals
    return out


def bootstrap_ci(d: np.ndarray, n_boot: int = 10000, seed: int = 0) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(d), size=(n_boot, len(d)))
    means = d[idx].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def holm(pvals: list[float]) -> list[float]:
    order = np.argsort(pvals)
    m = len(pvals)
    adj = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (m - rank) * pvals[i]))
        adj[i] = running
    return adj.tolist()


def compare(table, ref: str, against: list[str]) -> list[dict]:
    rows = []
    for other in against:
        cases = sorted(set(table[ref]) & set(table[other]))
        for metric, higher_better in METRICS.items():
            a = np.array([table[ref][c][metric] for c in cases])
            b = np.array([table[other][c][metric] for c in cases])
            ok = ~(np.isnan(a) | np.isnan(b))
            d = a[ok] - b[ok]
            better = d > 0 if higher_better else d < 0
            p = float(wilcoxon(d).pvalue) if np.any(d != 0) else 1.0
            lo, hi = bootstrap_ci(d)
            rows.append({
                "ref": ref, "vs": other, "metric": metric, "n": int(ok.sum()),
                "mean_diff": round(float(d.mean()), 4),
                "ci95": [round(lo, 4), round(hi, 4)],
                "ref_better_n": int(better.sum()),
                "p_wilcoxon": p,
            })
    for r, p_adj in zip(rows, holm([r["p_wilcoxon"] for r in rows])):
        r["p_holm"] = p_adj
        r["p_wilcoxon"] = float(f"{r['p_wilcoxon']:.3g}")
        r["p_holm"] = float(f"{p_adj:.3g}")
    return rows


def self_test() -> None:
    rng = np.random.default_rng(1)
    base = rng.uniform(0.55, 0.7, 30)
    table = {"a": {}, "b": {}, "c": {}}
    for i, x in enumerate(base):
        for meth, shift in (("a", 0.05), ("b", 0.0), ("c", 0.0)):
            noise = rng.normal(0, 0.01) if meth != "c" else 0.0
            v = x + shift + noise
            table[meth][f"c{i}"] = {m: (v if METRICS[m] else 1 - v) for m in METRICS}
    rows = compare(table, "a", ["b"])
    dice = next(r for r in rows if r["metric"] == "dice_L")
    assert dice["mean_diff"] > 0.04 and dice["ci95"][0] > 0 and dice["p_holm"] < 1e-3, dice
    assert dice["ref_better_n"] >= 28, dice
    asd = next(r for r in rows if r["metric"] == "asd_L_mm")
    assert asd["ref_better_n"] >= 28, asd
    assert holm([0.01, 0.04, 0.03]) == [0.03, 0.06, 0.06]
    print("self-test passed")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("table", nargs="?", type=Path)
    ap.add_argument("--ref", default="nnunet")
    ap.add_argument("--against", nargs="+", default=["maper", "jlf"])
    ap.add_argument("--out", type=Path)
    ap.add_argument("--self-test", action="store_true")
    a = ap.parse_args()
    if a.self_test:
        self_test()
        return
    if a.table is None:
        ap.error("table is required unless --self-test")
    rows = compare(load(a.table), a.ref, a.against)
    for r in rows:
        print(f"{r['ref']} vs {r['vs']:6s} {r['metric']:14s} n={r['n']:2d} "
              f"diff={r['mean_diff']:+.4f} CI95=[{r['ci95'][0]:+.4f},{r['ci95'][1]:+.4f}] "
              f"better {r['ref_better_n']:2d}/{r['n']} p_holm={r['p_holm']:.3g}")
    if a.out:
        a.out.write_text(json.dumps(rows, indent=1))


if __name__ == "__main__":
    main()
