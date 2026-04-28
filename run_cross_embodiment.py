"""Cross-embodiment eval: trained-on-Panda policies evaluated on Sawyer.

Thin wrapper that just calls run_calibration_sweep.py with --target_robot. Same
caveat as that script: live Shadow rendering at eval is gated on the
`lib/mj_silhouette.py` mujoco-direct path (TODO). Until that lands this is
useful for the RGB baseline cross-embodiment number (which the paper says is
near-zero — confirms the failure mode that Shadow is designed to fix).
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpts", nargs="+", required=True)
    p.add_argument("--target_robot", default="Sawyer")
    p.add_argument("--n_episodes", type=int, default=25)
    p.add_argument("--horizon", type=int, default=200)
    args = p.parse_args()

    cmd = [
        sys.executable, str(REPO / "run_calibration_sweep.py"),
        "--ckpts", *args.ckpts,
        "--target_robot", args.target_robot,
        "--n_episodes", str(args.n_episodes),
        "--horizon", str(args.horizon),
        "--out", str(REPO / "results" / "cross_embodiment.jsonl"),
    ]
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
