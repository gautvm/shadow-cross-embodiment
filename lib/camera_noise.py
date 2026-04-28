"""Camera-extrinsic noise.

Used in two places:
  - Train time (Shadow + noise variant): perturb the camera pose used to *render*
    the target-robot mask, so the policy learns to tolerate mask misalignment.
  - Eval time (calibration sweep): perturb the camera pose used to render the
    overlaid mask at deployment, simulating real-world calibration error.

The "true" camera observation that the env produces is left untouched — only
the camera used for the Shadow overlay rendering is perturbed.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation as R


def sample_extrinsic_noise(
    sigma_xyz: float,
    sigma_rot_deg: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Return a 4x4 noise transform to compose with the true camera-to-world pose.

    Args:
        sigma_xyz: stddev of zero-mean Gaussian translation noise (meters).
        sigma_rot_deg: stddev of zero-mean Gaussian rotation noise (degrees),
            applied as a random axis-angle perturbation.

    The returned matrix `T_noise` is meant to be left-multiplied into the true
    camera pose: `T_perturbed = T_noise @ T_true`.
    """
    dx = rng.normal(0.0, sigma_xyz, size=3) if sigma_xyz > 0 else np.zeros(3)
    if sigma_rot_deg > 0:
        # random axis, magnitude sampled from N(0, sigma_rot_deg)
        axis = rng.normal(size=3)
        axis = axis / (np.linalg.norm(axis) + 1e-12)
        angle_rad = np.deg2rad(rng.normal(0.0, sigma_rot_deg))
        rot = R.from_rotvec(axis * angle_rad).as_matrix()
    else:
        rot = np.eye(3)
    T = np.eye(4)
    T[:3, :3] = rot
    T[:3, 3] = dx
    return T


# Noise levels swept at evaluation. (sigma_xyz_meters, sigma_rot_degrees)
EVAL_NOISE_LEVELS = [
    (0.000, 0.0),
    (0.005, 2.0),
    (0.010, 5.0),
    (0.020, 10.0),
    (0.040, 20.0),
]
