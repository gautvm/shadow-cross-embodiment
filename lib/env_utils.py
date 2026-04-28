"""Env helpers shared across training, evaluation, and dataset generation.

Two robosuite-1.5 / mimicgen-1.0 compatibility quirks live here so they don't get
re-discovered in every script:

  1. Mimicgen datasets (env_version 1.4.1) store the legacy `OSC_POSE` controller
     config, which the new composite-controller system rejects.
  2. robomimic's ObsUtils registry is process-global state that must be initialized
     before any `env.reset()` / `env.get_observation()`.
"""

from __future__ import annotations

import robomimic.utils.env_utils as EnvUtils
import robomimic.utils.file_utils as FileUtils
import robomimic.utils.obs_utils as ObsUtils
from robosuite.controllers import load_composite_controller_config


# The set of obs keys we use across the project. RGB inputs differ per variant
# (rgb / shadow_vanilla / shadow_noise) but the low-dim proprio + object are common.
LOW_DIM_OBS = ["robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos", "object"]
RGB_OBS_DEFAULT = ["agentview_image"]   # we drop the wrist cam to keep the model small


def init_obs_utils(rgb_keys=None):
    rgb_keys = rgb_keys if rgb_keys is not None else RGB_OBS_DEFAULT
    ObsUtils.initialize_obs_utils_with_obs_specs(obs_modality_specs={
        "obs": {
            "low_dim": LOW_DIM_OBS,
            "rgb": rgb_keys,
        },
    })


def load_env_meta(dataset_path: str, robot: str = "Panda"):
    """Load env metadata from a dataset HDF5 and translate the controller config to v1.5."""
    env_meta = FileUtils.get_env_metadata_from_dataset(dataset_path)
    env_meta["env_kwargs"]["controller_configs"] = load_composite_controller_config(
        controller="BASIC", robot=robot,
    )
    return env_meta


def make_env(
    dataset_path: str,
    robot: str = "Panda",
    render: bool = False,
    render_offscreen: bool = True,
    use_image_obs: bool = True,
    camera_height: int = 84,
    camera_width: int = 84,
):
    """Make a robomimic-wrapped robosuite env from a dataset's env_meta, optionally swapping the robot."""
    env_meta = load_env_meta(dataset_path, robot=robot)
    env_meta["env_kwargs"]["robots"] = robot
    env_meta["env_kwargs"]["camera_heights"] = camera_height
    env_meta["env_kwargs"]["camera_widths"] = camera_width
    env = EnvUtils.create_env_from_metadata(
        env_meta=env_meta,
        render=render,
        render_offscreen=render_offscreen,
        use_image_obs=use_image_obs,
    )
    return env
