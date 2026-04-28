"""Phase 6 driver: sweep calibration-noise levels and policies, log success rates.

For each (policy, noise_level, seed) tuple, evaluate `n_episodes` rollouts and
write one JSON line per run to `results/calibration_sweep.jsonl`.

NOTE: live Shadow rendering during rollouts (the actual cal-noise injection at
eval) needs a mujoco-direct silhouette path that doesn't share a GL context
with the rollout env. That path is implemented in `lib/mj_silhouette.py` (TODO
— writing this is the remaining engineering before the headline plot).
Until that lands, this script supports `--shadow_edit none`, which gives:
  - the RGB baseline curve (correct — it's what it was trained on)
  - a sanity-check curve for Shadow-trained policies *without* the eval-time
    edit (will under-perform; useful as an upper-bound on what live edit
    needs to recover).
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent
RESULTS_DIR = REPO / "results"
SWEEP_OUT = RESULTS_DIR / "calibration_sweep.jsonl"

# (sigma_xyz [m], sigma_rot [deg]) — matches paper Table 3 spread.
NOISE_LEVELS = [
    (0.000, 0.0),
    (0.005, 2.0),
    (0.010, 5.0),
    (0.020, 10.0),
    (0.040, 20.0),
]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpts", nargs="+", required=True,
                   help="Paths to trained .pth checkpoints (one per policy variant × seed).")
    p.add_argument("--target_robot", default=None,
                   help="If set, do cross-embodiment eval instead of same-robot.")
    p.add_argument("--n_episodes", type=int, default=25)
    p.add_argument("--horizon", type=int, default=200)
    p.add_argument("--out", type=Path, default=SWEEP_OUT)
    p.add_argument("--shadow_edit", choices=["none", "precomputed"], default="none")
    args = p.parse_args()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    print(f"sweeping {len(args.ckpts)} ckpts × {len(NOISE_LEVELS)} noise levels  -> {args.out}")

    for ckpt in args.ckpts:
        for sx, sr in NOISE_LEVELS:
            tag = f"{Path(ckpt).stem}_sx{sx}_sr{sr}_target{args.target_robot or 'src'}"
            print(f"\n=== {tag} ===")
            cmd = [
                sys.executable, str(REPO / "evaluate.py"),
                "--ckpt", str(ckpt),
                "--n_episodes", str(args.n_episodes),
                "--horizon", str(args.horizon),
                "--shadow_edit", args.shadow_edit,
                "--out", str(args.out),
            ]
            if args.target_robot:
                cmd += ["--target_robot", args.target_robot]
            # NOTE: we run each (ckpt, noise) in a fresh subprocess so the env
            # GL context can't leak between runs.
            subprocess.run(cmd, check=True)

    print(f"\ndone. results -> {args.out}")


if __name__ == "__main__":
    main()
