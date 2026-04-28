# CLAUDE.md — Shadow reproduction project

This repo is an independent reproduction + extension of **SHADOW** (Lepert, Doshi, Bohg, CoRL 2024) intended as a research-quality demo for Gautam Paranjape's application to Stanford's IPRL lab. Read this before starting any work.

## What Shadow actually is — read carefully

Shadow's data edit is **composite**, not a full segmentation map:
1. Blackout the *source* robot's pixels in the image.
2. Render the *target* robot at the same end-effector pose using URDF + IK + camera params, and overlay its silhouette on top.
3. Background, table, object: preserved as raw RGB.

The "replace the whole image with segmentation masks" interpretation is **the failing baseline** the paper ablates against (Table 1 — "Black-only": 0.02 success on UR5e, 0.0 on IIWA). Do not implement that and call it Shadow.

## Experiment we are running (decided 2026-04-27)

**Option A: calibration-noise robustness.**

Paper Appendix A.4 / Table 3 shows Shadow degrades sharply under camera calibration error. The paper's own future-work line: *"we could attempt to render Shadow more robust to camera calibration error via (modest) noise injection during training."* We do exactly that.

**Three training conditions:**
- RGB baseline (no Shadow data edit)
- Vanilla Shadow (paper's method, no calibration noise during training)
- Shadow + noise injection (ours: Gaussian noise on camera extrinsics during the overlay step)

**Eval:** sweep eval-time calibration noise levels `{0, 0.5cm/2°, 1cm/5°, 2cm/10°, 4cm/20°}` × cross-embodiment to Sawyer. 100 episodes per condition. 3 seeds.

## Key design choices

- **Task:** Stack (Mimicgen, 951 demos). Simplest task in the paper. *Not* Lift — Lift isn't in the paper.
- **Source robot:** Panda + Robotiq 2F85. Fall back to default Panda gripper if Robotiq integration is painful — start simple, swap later.
- **Target robot:** Sawyer + same gripper.
- **Action space:** OSC_POSE (operational space control). Required for cross-embodiment because joint-space actions don't transfer between robots with different DOF.
- **Policy:** Diffusion Policy (CNN variant) from robomimic v0.5.0. Hyperparameters from paper Table 5 — 84×84, batch 256, lr 1e-4, 5000 epochs, To=2, Ta=8, horizon 200.
- **Compute:** Mac for development + evaluation. Cloud GPU (Colab Pro or GCP $300 free credit) for training only.

## Repo conventions

- Top-level `.py` files are entrypoints (`train.py`, `evaluate.py`, …). Library code in `lib/`.
- No notebooks. Plain `.py` only.
- Comments only where the *why* is non-obvious. Don't narrate what the code does.
- Plots: matplotlib + seaborn, with proper axis labels, legends, error bars, consistent style across figures.
- README is the artifact. It should always read like a mini research report, even mid-build.

## What NOT to do

- Don't implement "full image segmentation" and call it Shadow.
- Don't run a "camera viewpoint sweep" (rotating camera angle 5°→60°). It tests viewpoint invariance, not embodiment invariance — different paper. Originally proposed by Gautam, dropped on 2026-04-27.
- Don't use the Lift task. Use Stack.
- Don't pre-emptively add try/except blocks, abstract base classes, or config managers. Trust internal code; only validate at system boundaries.

## Local environment patches (already applied 2026-04-27)

The vendored repos in `~/Desktop/projects/{robosuite,robomimic,mimicgen}` carry two manual patches. If you re-clone them, re-apply:

1. **robomimic egl_probe.** Remove `"egl_probe>=1.0.1",` from `robomimic/setup.py` `install_requires`. In `robomimic/envs/env_robosuite.py`, wrap `import egl_probe` + `valid_gpu_devices = ...` in `try/except ImportError: pass`. Reason: egl_probe is Linux-only, needs cmake+make, breaks Mac install.
2. **MuJoCo pinned to 3.2.7.** robosuite v1.5.1 declares `>=3.2.3`, but mujoco 3.8.x rejects Sawyer/IIWA mesh inertia at load time. The `mink` dep wants `>=3.3.6` — that's a benign warning since we don't use GR1.

## Local environment patches (already applied 2026-04-27)

In addition to the egl_probe and mujoco patches above, two more patches were needed during Phase 2:

3. **mimicgen ↔ robosuite v1.5 single_arm_env.** Mimicgen 1.0 imports `robosuite.environments.manipulation.single_arm_env.SingleArmEnv`; that module was renamed to `manipulation_env.ManipulationEnv` in robosuite 1.5. Compat shim added: `robosuite/environments/manipulation/single_arm_env.py` re-exports `ManipulationEnv as SingleArmEnv`. One file, one line.
4. **Mimicgen dataset env_kwargs controller_configs.** Datasets store the legacy `OSC_POSE` controller config; robosuite v1.5 needs a composite controller. Translation lives in `lib/env_utils.load_env_meta` and in the training config under `experiment.env_meta_update_dict`.

## Critical rendering constraint (DO NOT FORGET)

**Two robosuite offscreen-render envs in the same process corrupt each other's segmentation rendering** (mujoco GL context interaction). All Shadow rendering must be done one env at a time:

- Offline (dataset generation): use the **two-pass design** in `lib/render_shadow.RobotMaskRenderer` + `create_shadow_dataset.py`. Pass 1 renders all real-robot masks with just the source-robot env; pass 2 destroys it, creates the target-robot env, renders all silhouettes via IK.
- Online (eval rollouts): the rollout env and the silhouette-overlay env can't both be alive. Live Shadow rendering during eval needs a **mujoco-direct silhouette path** (`lib/mj_silhouette.py` — TODO) that loads MJCF outside robosuite's wrapper and renders without sharing GL state. Until that lands, eval supports `--shadow_edit none` only.

## Status

- Phase 0 (scaffold): done.
- Phase 1 (local Mac env): done. Stack env loads on Panda and Sawyer; offscreen rendering works.
- Phase 2 (Mimicgen Stack data + RGB baseline): done. Dataset downloaded; train.py + base config wired.
- Phase 3 (Shadow data-edit): done. Two-pass renderer verified end-to-end on a 2-demo smoke (~28% masked, stable across timesteps).
- Phase 4 (camera-noise injection): done.
- Phase 5 (cloud GPU training): pending — launch via `scripts/launch_colab.ipynb`.
- Phase 6 (calibration sweep eval): scripts written; live Shadow eval gated on `lib/mj_silhouette.py`.
- Phase 7 (plots, README): plot scaffold ready; will populate once eval lands.
- Phase 8 (email to Ria): drafted last.
