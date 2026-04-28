"""Shadow's data edit, two-pass version.

Background: when two robosuite offscreen-render envs are alive in the same
process, the second one corrupts the first one's segmentation rendering (mujoco
GL context interaction). So we never keep both alive. Pass 1 builds all the
"real-robot" masks with just the real env. Pass 2 builds all the "virtual-robot"
silhouettes with just the virtual env (after IK). The composite happens after.

Public API:

    RobotMaskRenderer(robot, env_name, camera, height, width)
        .real_mask(state, arm_qpos, gripper_qpos) -> (H, W) bool
        .virt_mask(target_pos, target_quat_wxyz)   -> (H, W) bool

    composite(base_image, real_mask, virt_mask) -> (H, W, 3) uint8
"""

from __future__ import annotations

import logging
from typing import Optional

import mujoco
import numpy as np
import robosuite as suite
from robosuite.controllers import load_composite_controller_config
from scipy.spatial.transform import Rotation as R

logging.disable(logging.WARNING)

FILL_COLOR = np.array([0, 0, 0], dtype=np.uint8)   # paper uses solid black

_ROBOT_BODY_PREFIXES = ("robot0_", "gripper0_", "fixed_mount0_", "mount0_")


def _make_seg_env(robot: str, env_name: str, camera_name: str,
                  height: int, width: int):
    cfg = load_composite_controller_config(controller="BASIC", robot=robot)
    return suite.make(
        env_name, robots=robot, controller_configs=cfg,
        has_renderer=False, has_offscreen_renderer=True, use_camera_obs=True,
        camera_names=[camera_name],
        camera_segmentations=["element"],
        camera_heights=height, camera_widths=width,
        control_freq=20, ignore_done=True,
    )


def _robot_geom_ids(env) -> np.ndarray:
    """Geom IDs that belong to the robot, gripper, or fixed mount, by body name."""
    m = env.sim.model._model
    out = []
    for gid in range(m.ngeom):
        body_id = int(m.geom_bodyid[gid])
        bname = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, body_id) or ""
        if any(bname.startswith(p) for p in _ROBOT_BODY_PREFIXES):
            out.append(gid)
    return np.asarray(out, dtype=np.int32)


def _solve_ik_mujoco(
    sim,
    target_pos: np.ndarray,
    target_quat_wxyz: np.ndarray,
    eef_body_name: str,
    qpos_idx: np.ndarray,
    n_iter: int = 200,
    tol: float = 1e-3,
    damping: float = 5e-2,
    step: float = 0.5,
) -> np.ndarray:
    """Damped least-squares IK on mujoco's mjData. Mutates sim.data.qpos at qpos_idx."""
    model = sim.model._model
    data = sim.data._data
    body_id = model.body(eef_body_name).id
    target_R = R.from_quat([
        target_quat_wxyz[1], target_quat_wxyz[2], target_quat_wxyz[3], target_quat_wxyz[0]
    ]).as_matrix()
    jacp = np.zeros((3, model.nv))
    jacr = np.zeros((3, model.nv))
    for _ in range(n_iter):
        mujoco.mj_forward(model, data)
        cur_pos = data.body(body_id).xpos.copy()
        cur_R = data.body(body_id).xmat.reshape(3, 3).copy()
        err_pos = target_pos - cur_pos
        err_rotvec = R.from_matrix(target_R @ cur_R.T).as_rotvec()
        err = np.concatenate([err_pos, err_rotvec])
        if np.linalg.norm(err) < tol:
            break
        mujoco.mj_jacBody(model, data, jacp, jacr, body_id)
        J = np.vstack([jacp, jacr])[:, qpos_idx]
        dq = J.T @ np.linalg.solve(J @ J.T + damping * np.eye(6), err)
        data.qpos[qpos_idx] += step * dq
    return data.qpos[qpos_idx].copy()


class RobotMaskRenderer:
    """Renders a (H, W) bool mask of a robot's pixels.

    Two ways to query:
      - `real_mask(state, arm_qpos, gripper_qpos)`: drives the env to a stored
        dataset state and returns the robot pixel mask.
      - `virt_mask(target_pos, target_quat_wxyz)`: solves IK to put the robot's
        EE at the given pose and returns the robot pixel mask. Object/scene
        positions are wherever they were after env.reset() — fine, since we only
        keep the robot's pixels.
    """

    def __init__(self, robot: str, env_name: str = "Stack",
                 camera_name: str = "agentview", height: int = 84, width: int = 84):
        self.robot = robot
        self.camera_name = camera_name
        self.env = _make_seg_env(robot, env_name, camera_name, height, width)
        self.env.reset()
        self.robot_geom_ids = _robot_geom_ids(self.env)
        self._arm_idx = np.asarray(self.env.robots[0]._ref_joint_pos_indexes)
        gi = self.env.robots[0]._ref_gripper_joint_pos_indexes
        if isinstance(gi, dict):
            gi = sum((list(v) for v in gi.values()), [])
        self._gripper_idx = np.asarray(gi, dtype=np.int64)
        eef = self.env.robots[0].robot_model.eef_name
        self._eef_body = eef["right"] if isinstance(eef, dict) else eef

    def _seg_and_image(self):
        """Single observation refresh — returns (seg, rgb_image) so callers don't
        re-render. The rgb_image here is what the seg corresponds to pixel-for-pixel."""
        obs = self.env._get_observations(force_update=True)
        seg = np.asarray(obs[f"{self.camera_name}_segmentation_element"]).squeeze().astype(np.int32)
        img = np.asarray(obs[f"{self.camera_name}_image"])
        self._last_image = img
        return seg, img

    def _seg(self) -> np.ndarray:
        seg, _ = self._seg_and_image()
        return seg

    def _mask_from_seg(self, seg: np.ndarray) -> np.ndarray:
        return np.isin(seg, self.robot_geom_ids)

    def real_mask(self, state: Optional[np.ndarray], arm_qpos: np.ndarray,
                  gripper_qpos: np.ndarray) -> np.ndarray:
        sim = self.env.sim
        if state is not None:
            sim.set_state_from_flattened(state)
        sim.data.qpos[self._arm_idx] = arm_qpos
        if len(self._gripper_idx) and len(gripper_qpos):
            sim.data.qpos[self._gripper_idx] = gripper_qpos[: len(self._gripper_idx)]
        sim.forward()
        return self._mask_from_seg(self._seg())

    def virt_mask(self, target_pos: np.ndarray, target_quat_wxyz: np.ndarray) -> np.ndarray:
        _solve_ik_mujoco(
            self.env.sim,
            target_pos=target_pos,
            target_quat_wxyz=target_quat_wxyz,
            eef_body_name=self._eef_body,
            qpos_idx=self._arm_idx,
        )
        return self._mask_from_seg(self._seg())

    def get_image(self) -> np.ndarray:
        # returns the image cached from the last _seg_and_image() / real_mask() / virt_mask() call.
        return self._last_image


def composite(base_image: np.ndarray, real_mask: np.ndarray, virt_mask: np.ndarray,
              fill: np.ndarray = FILL_COLOR) -> np.ndarray:
    """Apply the Shadow data edit: blackout real-robot pixels + overlay virtual silhouette."""
    out = base_image.copy()
    out[real_mask] = fill
    out[virt_mask] = fill
    return out


def quat_xyzw_to_wxyz(q):
    q = np.asarray(q)
    return np.array([q[3], q[0], q[1], q[2]])
