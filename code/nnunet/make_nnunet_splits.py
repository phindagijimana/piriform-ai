#!/usr/bin/env python3
"""Write nnU-Net splits_final.json.

Stage A: copy the locked Hammers 5-fold (24/6 per fold).
Later (ADNI/Bonn): --stratify groups by name prefix so each fold mixes sites.
"""
from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

import os

ROOT = Path("/path/to/piriform")
DATASET = os.environ.get("NNUNET_DATASET", "Dataset001_Piriform")
LOCKED = ROOT / "nnunet/splits" / f"{DATASET}.json"
LABELS = ROOT / "nnunet/nnUNet_raw" / DATASET / "labelsTr"
PREPROC = ROOT / "nnunet/nnUNet_preprocessed" / DATASET


def case_ids() -> list[str]:
    return sorted(p.stem.replace(".nii", "") for p in LABELS.glob("*.nii.gz"))


def prefix(case: str) -> str:
    return case.split("_", 1)[0]


def stratified_folds(cases: list[str], n_folds: int, seed: int) -> list[dict]:
    by = defaultdict(list)
    for c in cases:
        by[prefix(c)].append(c)
    rng = random.Random(seed)
    for group in by.values():
        rng.shuffle(group)
    folds = [{"train": [], "val": []} for _ in range(n_folds)]
    for group in by.values():
        for i, c in enumerate(group):
            folds[i % n_folds]["val"].append(c)
    all_cases = set(cases)
    for fold in folds:
        val = set(fold["val"])
        fold["train"] = sorted(all_cases - val)
        fold["val"] = sorted(fold["val"])
    return folds


def write_splits(splits: list[dict]) -> None:
    PREPROC.mkdir(parents=True, exist_ok=True)
    text = json.dumps(splits, indent=4) + "\n"
    (PREPROC / "splits_final.json").write_text(text)
    LOCKED.parent.mkdir(parents=True, exist_ok=True)
    LOCKED.write_text(text)
    print(f"wrote {len(splits)} folds -> {PREPROC / 'splits_final.json'}")
    for i, fold in enumerate(splits):
        print(f"  fold {i}: train={len(fold['train'])} val={len(fold['val'])} {fold['val']}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--stratify", action="store_true",
                   help="Rebuild 5-fold stratified by Hammers_/ADNI_/Bonn_ prefix")
    p.add_argument("--seed", type=int, default=2023)
    args = p.parse_args()
    cases = case_ids()
    if not cases:
        raise SystemExit(f"no labels in {LABELS}")
    prefixes = sorted({prefix(c) for c in cases})
    if LOCKED.is_file() and not args.stratify:
        splits = json.loads(LOCKED.read_text())
        write_splits(splits)
        print(f"locked {DATASET} splits (not reshuffled)")
        return
    if args.stratify or len(prefixes) > 1:
        print(f"stratified splits for prefixes {prefixes}")
        write_splits(stratified_folds(cases, 5, args.seed))
        return
    write_splits(stratified_folds(cases, 5, args.seed))


if __name__ == "__main__":
    main()
