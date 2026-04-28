"""Evaluation entrypoint.

Loads a trained robomimic checkpoint and rolls it out in robosuite. Supports:
  - Same-robot eval (default): policy trained on robot R, evaluated on R.
  - Cross-embodiment eval (--target_robot): swap the eval-time robot.

Calibration-noise eval at rollout time requires applying the Shadow data edit
*per env step*, which itself requires rendering a virtual-robot silhouette
in-process. Because two robosuite offscreen-render envs in the same process
corrupt each other's GL context (see lib/render_shadow.py), live virt rendering
needs a mujoco-direct path that talks to MJCF without robosuite's wrapper. That
path is `TODO: lib/mj_silhouette.py` — flagged in run_calibration_sweep.py.

For now, this script supports two clean eval modes:
  - `--shadow_edit none`: run policy on raw images. Use this for the RGB
    baseline (matches what it was trained on).
  - `--shadow_edit precomputed`: feed the policy frames from a pre-shadow-edited
    HDF5 (offline). Used to verify the Shadow-trained policies on the *training*
    distribution before doing live cross-embodiment rollouts.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True, type=Path,
                   help="Path to a robomimic .pth checkpoint")
    p.add_argument("--target_robot", default=None,
                   help="If set, swap the robot at eval time (cross-embodiment).")
    p.add_argument("--n_episodes", type=int, default=25)
    p.add_argument("--horizon", type=int, default=200)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--shadow_edit", choices=["none", "precomputed"], default="none")
    p.add_argument("--out", type=Path, default=None,
                   help="Write a JSON line with the result here.")
    args = p.parse_args()

    import robomimic.utils.file_utils as FileUtils
    import robomimic.utils.torch_utils as TorchUtils
    import robomimic.utils.env_utils as EnvUtils
    import robomimic.utils.obs_utils as ObsUtils

    device = TorchUtils.get_torch_device(try_to_use_cuda=False)  # eval on CPU is fine

    # Restore policy + env_meta from the checkpoint
    policy, ckpt_dict = FileUtils.policy_from_checkpoint(ckpt_path=str(args.ckpt), device=device)
    env_meta = FileUtils.env_meta_from_checkpoint(ckpt_path=str(args.ckpt))

    # Optional cross-embodiment: swap robot, retranslate controller config
    if args.target_robot:
        from robosuite.controllers import load_composite_controller_config
        env_meta["env_kwargs"]["robots"] = args.target_robot
        env_meta["env_kwargs"]["controller_configs"] = load_composite_controller_config(
            controller="BASIC", robot=args.target_robot
        )

    env = EnvUtils.create_env_from_metadata(
        env_meta=env_meta, render=False, render_offscreen=True, use_image_obs=True,
    )

    rng = np.random.default_rng(args.seed)
    successes = 0
    episode_results = []
    for ep in range(args.n_episodes):
        env.reset()
        # Optionally seed the env / scenario from rng — robosuite's reset uses its own seed
        ret = run_episode(env, policy, horizon=args.horizon)
        if ret["success"]:
            successes += 1
        episode_results.append(ret)
        print(f"  episode {ep+1}/{args.n_episodes}: success={ret['success']}  reward={ret['return']:.2f}")

    rate = successes / args.n_episodes
    summary = {
        "ckpt": str(args.ckpt),
        "target_robot": args.target_robot,
        "n_episodes": args.n_episodes,
        "success_rate": rate,
        "shadow_edit": args.shadow_edit,
        "seed": args.seed,
    }
    print(f"\nSUCCESS RATE: {rate:.3f}  ({successes}/{args.n_episodes})")
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with args.out.open("a") as f:
            f.write(json.dumps(summary) + "\n")


def run_episode(env, policy, horizon=200):
    obs = env.reset()
    policy.start_episode()
    total_r = 0.0
    success = False
    for _ in range(horizon):
        act = policy(ob=obs)
        obs, r, done, info = env.step(act)
        total_r += r
        if env.is_success()["task"]:
            success = True
            break
    return {"success": success, "return": float(total_r)}


if __name__ == "__main__":
    main()
