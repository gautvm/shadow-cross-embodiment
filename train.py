"""Training entrypoint.

Thin wrapper around robomimic's `train` script. Picks the right dataset path
for the variant, names the run, sets the seed, and hands control to robomimic.

Usage:
    python train.py --variant rgb              --seed 0
    python train.py --variant shadow_vanilla   --seed 0
    python train.py --variant shadow_noise     --seed 0
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

REPO = Path(__file__).resolve().parent

DATASET_FOR_VARIANT = {
    "rgb":             REPO / "datasets" / "core" / "stack_d0.hdf5",
    "shadow_vanilla":  REPO / "datasets" / "shadow" / "stack_d0_shadow.hdf5",
    "shadow_noise":    REPO / "datasets" / "shadow" / "stack_d0_shadow_noise.hdf5",
}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--variant", required=True, choices=list(DATASET_FOR_VARIANT))
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--epochs", type=int, default=None,
                   help="override num_epochs (useful for tiny smoke runs)")
    p.add_argument("--batch_size", type=int, default=None)
    p.add_argument("--no_cuda", action="store_true",
                   help="force CPU (Mac smoke run)")
    p.add_argument("--config_out", type=str, default=None,
                   help="if set, write the resolved config here and exit (no training)")
    args = p.parse_args()

    base_path = REPO / "configs" / "base.json"
    cfg = json.loads(base_path.read_text())

    cfg["train"]["data"] = str(DATASET_FOR_VARIANT[args.variant])
    cfg["train"]["seed"] = args.seed
    cfg["train"]["cuda"] = not args.no_cuda
    cfg["experiment"]["name"] = f"{args.variant}_seed{args.seed}"

    # Resolve the v1.5 composite controller config at runtime — its shape
    # changed between robosuite versions, so we let robosuite produce it.
    from robosuite.controllers import load_composite_controller_config
    cfg["experiment"]["env_meta_update_dict"] = {
        "env_kwargs": {
            "controller_configs": load_composite_controller_config(controller="BASIC", robot="Panda"),
        },
    }
    if args.epochs is not None:
        cfg["train"]["num_epochs"] = args.epochs
        # Only shrink steps-per-epoch for tiny smoke runs; real runs keep paper-matched 100.
        if args.epochs <= 10:
            cfg["experiment"]["epoch_every_n_steps"] = 50
    if args.batch_size is not None:
        cfg["train"]["batch_size"] = args.batch_size

    out_dir = REPO / cfg["train"]["output_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)
    resolved_cfg_path = out_dir / f"_resolved_{args.variant}_seed{args.seed}.json"
    resolved_cfg_path.write_text(json.dumps(cfg, indent=2))
    print(f"resolved config -> {resolved_cfg_path}")

    if args.config_out:
        Path(args.config_out).write_text(json.dumps(cfg, indent=2))
        print(f"wrote config to {args.config_out} (no training)")
        return

    # Hand off to robomimic's training entrypoint as a subprocess
    # (its parse_args isn't exposed as a function — it's inline under __main__).
    import subprocess, sys
    subprocess.run(
        [sys.executable, "-m", "robomimic.scripts.train", "--config", str(resolved_cfg_path)],
        check=True,
    )


if __name__ == "__main__":
    main()
