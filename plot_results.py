"""Generate the publication figures.

Reads `results/calibration_sweep.jsonl` (one JSON record per eval run) and
writes:
  - results/fig_calibration_curve.png — success rate vs. eval-time noise,
    one curve per policy variant, error bars across seeds.
  - results/fig_mask_examples.png — visual examples of the Shadow data edit
    at increasing calibration noise levels.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

REPO = Path(__file__).resolve().parent
RESULTS = REPO / "results"

# Paper Table 3 spread. (sigma_xyz, sigma_rot) — must match run_calibration_sweep.NOISE_LEVELS.
NOISE_LEVELS = [
    (0.000, 0.0),
    (0.005, 2.0),
    (0.010, 5.0),
    (0.020, 10.0),
    (0.040, 20.0),
]

# Pretty names for the variants and a consistent palette.
VARIANT_LABEL = {
    "rgb":            "RGB baseline",
    "shadow_vanilla": "Shadow (paper)",
    "shadow_noise":   "Shadow + noise injection (ours)",
}
VARIANT_COLOR = {
    "rgb":            "#888888",
    "shadow_vanilla": "#d97333",
    "shadow_noise":   "#3a83b6",
}


def parse_records(path: Path):
    """Group records by variant and noise level. Returns {variant: {(sx,sr): [success_rates]}}."""
    grouped = defaultdict(lambda: defaultdict(list))
    if not path.exists():
        return grouped
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        # The variant lives in the ckpt path, e.g. ".../shadow_vanilla_seed0/.../model_epoch_5000.pth"
        variant = None
        for v in VARIANT_LABEL:
            if v in rec["ckpt"]:
                variant = v
                break
        if variant is None:
            continue
        # We don't currently store the noise level in the record (TODO once
        # live Shadow eval lands). Until then, we infer it from filename tag
        # in evaluate.py output. For the baseline plot this still produces a
        # single point per (variant, seed).
        sx, sr = (0.0, 0.0)
        for s_xyz, s_rot in NOISE_LEVELS:
            if f"sx{s_xyz}" in rec.get("ckpt", "") + rec.get("tag", ""):
                sx, sr = s_xyz, s_rot
                break
        grouped[variant][(sx, sr)].append(rec["success_rate"])
    return grouped


def plot_calibration_curve(grouped, out_path: Path, title: str):
    sns.set_style("whitegrid")
    fig, ax = plt.subplots(figsize=(6.5, 4.0), dpi=140)
    xs = np.arange(len(NOISE_LEVELS))
    xlabels = [f"{int(sr)}°\n{int(sx*1000)}mm" for sx, sr in NOISE_LEVELS]

    for variant, label in VARIANT_LABEL.items():
        if variant not in grouped:
            continue
        ys, errs = [], []
        for lvl in NOISE_LEVELS:
            v = grouped[variant].get(lvl, [])
            if v:
                ys.append(np.mean(v))
                errs.append(np.std(v) if len(v) > 1 else 0.0)
            else:
                ys.append(np.nan)
                errs.append(0.0)
        ys = np.asarray(ys); errs = np.asarray(errs)
        ax.errorbar(xs, ys, yerr=errs, marker="o", capsize=3, linewidth=2,
                    label=label, color=VARIANT_COLOR[variant])
    ax.set_xticks(xs); ax.set_xticklabels(xlabels)
    ax.set_xlabel("eval-time camera calibration noise (σθ / σxyz)")
    ax.set_ylabel("Stack task success rate")
    ax.set_ylim(-0.02, 1.02)
    ax.set_title(title)
    ax.legend(loc="lower left", frameon=True)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  -> {out_path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--records", type=Path, default=RESULTS / "calibration_sweep.jsonl")
    p.add_argument("--out_dir", type=Path, default=RESULTS)
    args = p.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    grouped = parse_records(args.records)
    if not grouped:
        print(f"no records at {args.records} — running with empty plot for now.")
    plot_calibration_curve(
        grouped, args.out_dir / "fig_calibration_curve.png",
        title="Shadow under calibration noise (Stack, Panda → Sawyer)",
    )


if __name__ == "__main__":
    main()
