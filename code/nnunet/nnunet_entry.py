#!/usr/bin/env python3
"""Run nnU-Net with cuDNN off (NGC vLLM 26.02 + L40S MIG has no conv engines)."""
from __future__ import annotations

import sys

import torch

torch.backends.cudnn.enabled = False
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.allow_tf32 = False
torch.backends.cuda.matmul.allow_tf32 = False

COMMANDS = {
    "train": "nnunetv2.run.run_training:run_training_entry",
    "predict": "nnunetv2.inference.predict_from_raw_data:predict_entry_point",
    "find_best": "nnunetv2.evaluation.find_best_configuration:find_best_configuration_entry_point",
}


def main() -> None:
    if len(sys.argv) < 2 or sys.argv[1] not in COMMANDS:
        raise SystemExit("usage: nnunet_entry.py train|predict|find_best [nnU-Net args...]")
    print(
        f"nnunet_entry: cudnn.enabled={torch.backends.cudnn.enabled} "
        f"device={torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}",
        flush=True,
    )
    spec = COMMANDS[sys.argv[1]]
    mod_name, fn_name = spec.split(":")
    sys.argv = [sys.argv[0]] + sys.argv[2:]
    module = __import__(mod_name, fromlist=[fn_name])
    getattr(module, fn_name)()


if __name__ == "__main__":
    main()
