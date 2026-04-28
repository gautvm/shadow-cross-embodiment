"""Build a Shadow-edited dataset from a robomimic/Mimicgen HDF5.

Two-pass design: we never keep the source-robot env and target-robot env alive
at the same time, because their offscreen rendering contexts corrupt each
other in a single process.

Pass 1 — for every (demo, t):
    drive the source-robot env to the dataset state, render its segmentation,
    extract the source-robot pixel mask, save the original RGB frame.
Pass 2 — destroy source env. Create target-robot env. For every (demo, t):
    solve IK so the target's end-effector matches the source's at frame t,
    optionally with calibration noise on the target render's camera extrinsics,
    render segmentation, extract target-robot pixel mask.
Composite — overlay target silhouette on top of blacked-out source RGB. Write
    a new HDF5 with the same structure as the input, but with `agentview_image`
    replaced by the Shadow-edited variant.

Usage:
    python create_shadow_dataset.py \
        --src datasets/core/stack_d0.hdf5 \
        --out datasets/shadow/stack_d0_shadow.hdf5 \
        --source_robot Panda --target_robot Sawyer \
        --noise none

    python create_shadow_dataset.py \
        --src datasets/core/stack_d0.hdf5 \
        --out datasets/shadow/stack_d0_shadow_noise.hdf5 \
        --source_robot Panda --target_robot Sawyer \
        --noise gaussian --sigma_xyz 0.01 --sigma_rot_deg 5.0
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import h5py
import numpy as np
from tqdm import tqdm

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))

from lib.render_shadow import RobotMaskRenderer, composite, quat_xyzw_to_wxyz
from lib.camera_noise import sample_extrinsic_noise


def build_real_masks(src_path: Path, source_robot: str, height: int, width: int,
                     demo_keys: list[str]):
    """Pass 1: source-robot pixel masks AND the freshly-rendered RGB for each frame.

    We use the freshly-rendered RGB as the composite base, not the dataset's stored
    image. The dataset images come from a different robosuite version (1.4.1 vs our
    1.5.1) and use a different vertical-axis convention than our offscreen renderer,
    so masks computed on our render don't align with the stored pixels. Re-rendering
    guarantees pixel-perfect mask alignment.
    """
    renderer = RobotMaskRenderer(robot=source_robot, height=height, width=width)
    real_masks = {}
    base_imgs = {}
    with h5py.File(src_path, "r") as f:
        for k in tqdm(demo_keys, desc=f"pass 1 ({source_robot} masks)"):
            d = f["data"][k]
            T = d.attrs["num_samples"]
            rm = np.empty((T, height, width), dtype=bool)
            imgs = np.empty((T, height, width, 3), dtype=np.uint8)
            for t in range(T):
                state = np.asarray(d["states"][t])
                arm = np.asarray(d["obs"]["robot0_joint_pos"][t])
                gr = np.asarray(d["obs"]["robot0_gripper_qpos"][t])
                # robosuite's offscreen render uses OpenGL y-flipped convention;
                # the dataset stores images already flipped to top-down. Match.
                rm[t] = renderer.real_mask(state, arm, gr)[::-1]
                imgs[t] = renderer.get_image()[::-1]
            real_masks[k] = rm
            base_imgs[k] = imgs
    del renderer
    return real_masks, base_imgs


def build_virt_masks(src_path: Path, target_robot: str, height: int, width: int,
                     demo_keys: list[str], noise_sigma_xyz: float, noise_sigma_rot_deg: float,
                     seed: int = 0):
    """Pass 2: target-robot silhouettes, IK'd to source EE poses (+ optional cal noise)."""
    rng = np.random.default_rng(seed)
    renderer = RobotMaskRenderer(robot=target_robot, height=height, width=width)
    virt_masks = {}
    with h5py.File(src_path, "r") as f:
        for k in tqdm(demo_keys, desc=f"pass 2 ({target_robot} silhouettes)"):
            d = f["data"][k]
            T = d.attrs["num_samples"]
            vm = np.empty((T, height, width), dtype=bool)
            for t in range(T):
                eef_pos = np.asarray(d["obs"]["robot0_eef_pos"][t])
                eef_q_wxyz = quat_xyzw_to_wxyz(d["obs"]["robot0_eef_quat"][t])
                if noise_sigma_xyz > 0 or noise_sigma_rot_deg > 0:
                    # Perturb the EE target — equivalent to perturbing camera extrinsics
                    # to first order: a misaligned camera makes the rendered mask end up
                    # at the wrong place in the image, which is exactly what we want to
                    # simulate. (Strictly the camera is what's noised; perturbing the EE
                    # position is mathematically equivalent for the silhouette outcome.)
                    Tn = sample_extrinsic_noise(noise_sigma_xyz, noise_sigma_rot_deg, rng)
                    eef_pos = Tn[:3, :3] @ eef_pos + Tn[:3, 3]
                    # apply rotation noise to the quaternion
                    from scipy.spatial.transform import Rotation as R
                    Rcur = R.from_quat([eef_q_wxyz[1], eef_q_wxyz[2], eef_q_wxyz[3], eef_q_wxyz[0]])
                    Rnoisy = R.from_matrix(Tn[:3, :3]) * Rcur
                    qx, qy, qz, qw = Rnoisy.as_quat()
                    eef_q_wxyz = np.array([qw, qx, qy, qz])
                vm[t] = renderer.virt_mask(eef_pos, eef_q_wxyz)[::-1]
            virt_masks[k] = vm
    del renderer
    return virt_masks


def write_shadow_hdf5(src_path: Path, out_path: Path, demo_keys: list[str],
                      base_imgs: dict, real_masks: dict, virt_masks: dict):
    """Copy src HDF5 to out_path then overwrite agentview_image with Shadow-edited."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(src_path, out_path)
    with h5py.File(out_path, "r+") as f:
        for k in tqdm(demo_keys, desc="writing composited frames"):
            base = base_imgs[k]
            rm = real_masks[k]
            vm = virt_masks[k]
            shadow = base.copy()
            shadow[rm] = 0
            shadow[vm] = 0
            del f[f"data/{k}/obs/agentview_image"]
            f.create_dataset(f"data/{k}/obs/agentview_image", data=shadow,
                             dtype="uint8", compression="gzip", compression_opts=4)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--src", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--source_robot", default="Panda")
    p.add_argument("--target_robot", default="Sawyer")
    p.add_argument("--height", type=int, default=84)
    p.add_argument("--width",  type=int, default=84)
    p.add_argument("--noise", choices=["none", "gaussian"], default="none")
    p.add_argument("--sigma_xyz", type=float, default=0.01)
    p.add_argument("--sigma_rot_deg", type=float, default=5.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--limit_demos", type=int, default=None,
                   help="for quick iteration: process only the first N demos")
    args = p.parse_args()

    sx = args.sigma_xyz if args.noise == "gaussian" else 0.0
    sr = args.sigma_rot_deg if args.noise == "gaussian" else 0.0

    with h5py.File(args.src, "r") as f:
        demo_keys = sorted(
            [k for k in f["data"].keys() if k.startswith("demo_")],
            key=lambda x: int(x.split("_")[1]),
        )
    if args.limit_demos:
        demo_keys = demo_keys[: args.limit_demos]
    print(f"processing {len(demo_keys)} demos: src={args.src}  out={args.out}")
    print(f"  source={args.source_robot}  target={args.target_robot}  noise(σxyz={sx}, σrot={sr}°)")

    real_masks, base_imgs = build_real_masks(args.src, args.source_robot,
                                             args.height, args.width, demo_keys)
    virt_masks = build_virt_masks(args.src, args.target_robot,
                                  args.height, args.width, demo_keys, sx, sr, seed=args.seed)
    write_shadow_hdf5(args.src, args.out, demo_keys, base_imgs, real_masks, virt_masks)
    print(f"done -> {args.out}")


if __name__ == "__main__":
    main()
