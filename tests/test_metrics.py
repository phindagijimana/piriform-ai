"""Known-answer tests for the evaluation code. Synthetic arrays only."""
import math
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code" / "evaluation"))

from mauri_metrics import (  # noqa: E402
    apply_topology_fix,
    average_surface_distance_mm,
    dice_mask,
    topology_stats,
)
import paired_stats  # noqa: E402


def cube(shape=(20, 20, 20), lo=5, hi=10):
    m = np.zeros(shape, bool)
    m[lo:hi, lo:hi, lo:hi] = True
    return m


def test_dice_known_values():
    a = cube()
    assert dice_mask(a, a) == 1.0
    assert dice_mask(a, np.zeros_like(a)) == 0.0
    b = np.zeros_like(a)
    b[5:10, 5:10, 7:12] = True  # overlaps 3 of 5 slices
    assert math.isclose(dice_mask(a, b), 2 * 75 / 250)
    assert math.isnan(dice_mask(np.zeros_like(a), np.zeros_like(a)))


def test_asd_zero_for_identical_and_none_for_empty():
    a = cube()
    assert average_surface_distance_mm(a, a, np.array([1.0, 1.0, 1.0])) == 0.0
    assert average_surface_distance_mm(a, np.zeros_like(a), np.ones(3)) is None


def test_asd_grows_with_shift_and_voxel_size():
    a = cube()
    b = np.roll(a, 2, axis=0)
    d1 = average_surface_distance_mm(a, b, np.array([1.0, 1.0, 1.0]))
    d2 = average_surface_distance_mm(a, b, np.array([2.0, 2.0, 2.0]))
    assert d1 > 0
    assert math.isclose(d2, 2 * d1, rel_tol=1e-6)


def test_topology_six_connectivity():
    m = np.zeros((10, 10, 10), bool)
    m[2, 2, 2] = True
    m[3, 3, 2] = True  # diagonal neighbour: separate under 6-connectivity
    assert topology_stats(m)["n_cc"] == 2


def test_topology_counts_holes():
    m = cube(lo=3, hi=8)
    m[5, 5, 5] = False
    s = topology_stats(m)
    assert s == {"n_cc": 1, "n_holes_vox": 1, "largest_cc_frac": 1.0}


def test_topology_fix_keeps_largest_and_fills():
    data = np.zeros((20, 20, 20), np.uint8)
    data[3:9, 3:9, 3:9] = 1
    data[5, 5, 5] = 0  # hole
    data[15, 15, 15] = 1  # island
    data[12:14, 3:5, 3:5] = 2
    out = apply_topology_fix(data)
    assert out[15, 15, 15] == 0
    assert out[5, 5, 5] == 1
    assert (out == 2).sum() == 8


def test_paired_stats_self_test():
    paired_stats.self_test()


def test_score_hammers_self_test():
    r = subprocess.run(
        [sys.executable, str(ROOT / "code" / "evaluation" / "score_hammers_stage_a.py"), "--self-test"],
        capture_output=True, text=True, cwd=ROOT,
    )
    assert r.returncode == 0, r.stderr
