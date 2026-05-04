#!/usr/bin/env bash
# Provision a fresh Vast.ai instance for Shadow training.
# Mirrors the Kaggle notebook's install steps but as a single script.
#
# Run once per instance (idempotent — safe to re-run after restart).
# Expects to be run from /workspace.

set -euo pipefail

cd /workspace

echo "[setup] cloning vendored repos"
[ -d robosuite ] || git clone -q https://github.com/ARISE-Initiative/robosuite.git
[ -d robomimic ] || git clone -q https://github.com/ARISE-Initiative/robomimic.git
[ -d mimicgen  ] || git clone -q https://github.com/NVlabs/mimicgen.git
(cd robosuite && git checkout -q v1.5.1)

echo "[setup] applying patches"
# Patch 1: drop egl_probe (Linux-only build dep, doesn't affect runtime)
sed -i 's/.*egl_probe.*$//' robomimic/setup.py
python3 - <<'PY'
p = 'robomimic/robomimic/envs/env_robosuite.py'
s = open(p).read()
if 'try:\n                    import egl_probe' not in s:
    s = s.replace(
        'import egl_probe\n                valid_gpu_devices = egl_probe.get_available_devices()',
        'try:\n                    import egl_probe\n                    valid_gpu_devices = egl_probe.get_available_devices()\n                except ImportError:\n                    valid_gpu_devices = []')
    open(p, 'w').write(s)
PY

# Patch 2: SingleArmEnv compat shim for mimicgen against robosuite v1.5
cat > robosuite/robosuite/environments/manipulation/single_arm_env.py <<'PY'
from robosuite.environments.manipulation.manipulation_env import ManipulationEnv as SingleArmEnv  # noqa: F401
PY

echo "[setup] installing python deps"
pip install -q 'mujoco==3.2.7' cmake 'numpy<2.3' scipy seaborn h5py
pip install -q -e ./robosuite
pip install -q -e ./robomimic
pip install -q -e ./mimicgen

# Pin PyTorch to a build that supports the broadest range of GPUs we might rent.
# 2.4.1 + cu121 covers Pascal (sm_60) through Ada (sm_89). Newer 2.5+ drops sm_60.
pip install -q --force-reinstall --no-deps torch==2.4.1 torchvision==0.19.1 \
    --index-url https://download.pytorch.org/whl/cu121

# diffusers 0.11.1 detects flax (Kaggle/Linux GPU images often ship it) and
# tries to import scheduler files that break with newer JAX. Removing flax
# triggers diffusers' own dummy-stub fallback path. We use PyTorch only.
pip uninstall -y -q flax || true

# robosuite needs its macros file at first run
python3 robosuite/robosuite/scripts/setup_macros.py >/dev/null 2>&1 || true

echo "[setup] done"
