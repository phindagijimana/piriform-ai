#!/usr/bin/env python3
"""Train SynthSeg-PC via Billot GMM loop (no L/R flip).

Cites: Billot Nat Commun 2023; Mauri HBM 2025; Steinbart HBM 2023 (no flip).
Requires: clau_seg/tools/env python3.8 + SynthSeg on PYTHONPATH.

Recipe v2 (Mauri/Billot Dice levers): larger U-Net, richer GMM corruption,
longer dice phase — use a fresh model_dir (architecture ≠ fold0 dice_016).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DEFAULT_SYNTHSEG = Path(
    "/path/to/clau_seg/tools/SynthSeg"
)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--labels-dir", type=Path, default=ROOT / "data/labels_train")
    ap.add_argument("--model-dir", type=Path, default=ROOT / "models/fold0")
    ap.add_argument("--config", type=Path, default=ROOT / "config")
    ap.add_argument("--output-shape", type=int, default=160)
    ap.add_argument("--batchsize", type=int, default=1)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--wl2-epochs", type=int, default=1)
    ap.add_argument("--dice-epochs", type=int, default=50)
    ap.add_argument("--steps-per-epoch", type=int, default=1000)
    ap.add_argument("--checkpoint", type=str, default=None)
    ap.add_argument("--unet-feat-count", type=int, default=24)
    ap.add_argument("--scaling-bounds", type=float, default=0.2)
    ap.add_argument("--rotation-bounds", type=float, default=15.0)
    ap.add_argument("--shearing-bounds", type=float, default=0.012)
    ap.add_argument("--nonlin-std", type=float, default=4.0)
    ap.add_argument("--bias-field-std", type=float, default=0.7)
    ap.add_argument(
        "--recipe",
        type=str,
        default="v1",
        help="tag stored in train_run.json (v1|v2|wb)",
    )
    ap.add_argument(
        "--synthseg-root",
        type=Path,
        default=Path(os.environ.get("SYNTHSEG_ROOT", DEFAULT_SYNTHSEG)),
    )
    args = ap.parse_args()

    ss = args.synthseg_root.resolve()
    if not (ss / "SynthSeg" / "training.py").is_file():
        raise SystemExit(f"SynthSeg not found at {ss}")
    sys.path.insert(0, str(ss))

    gen = args.config / "generation_labels.npy"
    seg = args.config / "segmentation_labels.npy"
    cls = args.config / "generation_classes.npy"
    meta_p = args.config / "label_meta.json"
    for p in (gen, seg, cls, args.labels_dir):
        if not Path(p).exists():
            raise SystemExit(f"missing {p} — run prepare_labels.py first")

    meta = json.loads(meta_p.read_text()) if meta_p.is_file() else {}
    if meta.get("flipping", False):
        raise SystemExit("label_meta.flipping must be false for Steinbart PC")

    args.model_dir.mkdir(parents=True, exist_ok=True)
    run_meta = {
        "recipe": args.recipe,
        "labels_dir": str(args.labels_dir),
        "model_dir": str(args.model_dir),
        "flipping": False,
        "output_shape": args.output_shape,
        "wl2_epochs": args.wl2_epochs,
        "dice_epochs": args.dice_epochs,
        "steps_per_epoch": args.steps_per_epoch,
        "unet_feat_count": args.unet_feat_count,
        "scaling_bounds": args.scaling_bounds,
        "rotation_bounds": args.rotation_bounds,
        "shearing_bounds": args.shearing_bounds,
        "nonlin_std": args.nonlin_std,
        "bias_field_std": args.bias_field_std,
        "randomise_res": True,
        "label_meta": meta,
        "cite": meta.get("cite", ["Billot et al. Nat Commun 2023"]),
    }
    (args.model_dir / "train_run.json").write_text(json.dumps(run_meta, indent=2) + "\n")
    print(
        json.dumps(
            {
                k: run_meta[k]
                for k in (
                    "recipe",
                    "model_dir",
                    "flipping",
                    "dice_epochs",
                    "steps_per_epoch",
                    "unet_feat_count",
                    "scaling_bounds",
                    "bias_field_std",
                    "nonlin_std",
                )
            }
        )
    )

    from SynthSeg.training import training  # noqa: WPS433

    training(
        labels_dir=str(args.labels_dir),
        model_dir=str(args.model_dir),
        generation_labels=str(gen),
        segmentation_labels=str(seg),
        n_neutral_labels=None,  # flipping off — Steinbart laterality
        generation_classes=str(cls),
        batchsize=args.batchsize,
        n_channels=1,
        target_res=None,
        output_shape=args.output_shape,
        prior_distributions="uniform",
        flipping=False,
        scaling_bounds=args.scaling_bounds,
        rotation_bounds=args.rotation_bounds,
        shearing_bounds=args.shearing_bounds,
        translation_bounds=False,
        nonlin_std=args.nonlin_std,
        randomise_res=True,
        bias_field_std=args.bias_field_std,
        n_levels=5,
        nb_conv_per_level=2,
        conv_size=3,
        unet_feat_count=args.unet_feat_count,
        feat_multiplier=2,
        activation="elu",
        lr=args.lr,
        wl2_epochs=args.wl2_epochs,
        dice_epochs=args.dice_epochs,
        steps_per_epoch=args.steps_per_epoch,
        checkpoint=args.checkpoint,
    )


if __name__ == "__main__":
    main()
