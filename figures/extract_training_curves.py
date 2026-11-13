#!/usr/bin/env python3
"""Extract per-epoch training curves from nnU-Net and SynthSeg (Keras) logs.

Only scalar per-epoch values are kept (loss, pseudo-Dice, learning rate).
Subject IDs, paths, and hostnames in the logs are discarded, so the output
JSON is safe to commit.

Usage:
  python figures/extract_training_curves.py \
      --nnunet-results $nnUNet_results \
      --synthseg-v1 logs/synthseg-pc-<jobid>.out \
      --synthseg-v2 logs/synthseg-pc-v2-<jobid>.out
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

TRAINER = "nnUNetTrainerNoMirroring__nnUNetPlans__3d_fullres"
DATASETS = {
    "Dataset001_Piriform": "001 Hammers",
    "Dataset002_PiriformCrop60": "002 Hammers crop",
    "Dataset003_PiriformStageB": "003 Hammers+ADNI",
    "Dataset004_PiriformStageBHarm": "004 Hammers+ADNI harmonised",
    "Dataset005_PiriformADNI": "005 ADNI only",
}

RE_EPOCH = re.compile(r": Epoch (\d+)\s*$")
RE_LR = re.compile(r": Current learning rate: ([0-9.e-]+)")
RE_TRAIN = re.compile(r": train_loss (-?[0-9.]+)")
RE_VAL = re.compile(r": val_loss (-?[0-9.]+)")
RE_DICE = re.compile(r": Pseudo dice \[(.*)\]")
RE_NUM = re.compile(r"(?:np\.float32\()?(-?[0-9.]+)\)?")


def parse_nnunet_fold(fold_dir: Path) -> dict[str, list]:
    """Merge all training_log_*.txt of one fold; a resumed epoch overwrites the earlier one."""
    rows: dict[int, dict] = {}
    for log in sorted(fold_dir.glob("training_log_*.txt")):
        ep = None
        for line in log.read_text(errors="replace").splitlines():
            if m := RE_EPOCH.search(line):
                ep = int(m.group(1))
                rows[ep] = {}
            elif ep is None:
                continue
            elif m := RE_LR.search(line):
                rows[ep]["lr"] = float(m.group(1))
            elif m := RE_TRAIN.search(line):
                rows[ep]["train_loss"] = float(m.group(1))
            elif m := RE_VAL.search(line):
                rows[ep]["val_loss"] = float(m.group(1))
            elif m := RE_DICE.search(line):
                vals = [float(x) for x in RE_NUM.findall(m.group(1))]
                rows[ep]["pseudo_dice"] = sum(vals) / len(vals) if vals else None
    done = [e for e in sorted(rows) if "val_loss" in rows[e]]
    keys = ("lr", "train_loss", "val_loss", "pseudo_dice")

    def r(x):
        return None if x is None else round(x, 5)

    return {"epoch": done, **{k: [r(rows[e].get(k)) for e in done] for k in keys}}


RE_K_EPOCH = re.compile(r"^Epoch (\d+)/(\d+)")
RE_K_END = re.compile(r"^\s*(\d+)/\1 \[=+\] - \d+s .*loss: ([0-9.]+)")
RE_K_SAVE = re.compile(r"^Epoch \d+: saving model to .*/(wl2|dice)_(\d+)\.h5")


def parse_keras(log: Path) -> dict[str, list]:
    """Epoch-final training loss from a Keras progress log, tagged by phase (wl2 / dice)."""
    text = log.read_text(errors="replace").replace("\r", "\n")
    out = {"phase": [], "epoch": [], "loss": []}
    last_loss = None
    for line in text.splitlines():
        if m := RE_K_END.match(line):
            last_loss = float(m.group(2))
        elif (m := RE_K_SAVE.match(line)) and last_loss is not None:
            out["phase"].append(m.group(1))
            out["epoch"].append(int(m.group(2)))
            out["loss"].append(last_loss)
            last_loss = None
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--nnunet-results", type=Path, required=True)
    ap.add_argument("--synthseg-v1", type=Path)
    ap.add_argument("--synthseg-v2", type=Path)
    ap.add_argument("--out", type=Path, default=Path(__file__).parent / "data" / "training_curves.json")
    a = ap.parse_args()

    out: dict = {"nnunet": {}, "synthseg": {}}
    for ds, name in DATASETS.items():
        base = a.nnunet_results / ds / TRAINER
        folds = {p.name: parse_nnunet_fold(p) for p in sorted(base.glob("fold_*"))}
        folds = {k: v for k, v in folds.items() if v["epoch"]}
        if folds:
            out["nnunet"][name] = folds
    for tag, log in (("v1", a.synthseg_v1), ("v2", a.synthseg_v2)):
        if log:
            out["synthseg"][tag] = parse_keras(log)

    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(out, separators=(",", ":")))
    for name, folds in out["nnunet"].items():
        print(name, {k: len(v["epoch"]) for k, v in folds.items()})
    for tag, d in out["synthseg"].items():
        print("synthseg", tag, list(zip(d["phase"], d["epoch"])))


if __name__ == "__main__":
    main()
