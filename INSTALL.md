# Installation

Two environments: **local Mac** (dev, eval, plotting) and **cloud GPU** (training only). Same conda recipe; PyTorch flavor differs.

## Part 1 — Local Mac

### 1. Miniconda

If you don't already have conda:

```bash
brew install --cask miniconda
conda init zsh
# restart your terminal
```

### 2. Create the env

```bash
conda create -n shadow python=3.10 -y
conda activate shadow
```

(robomimic docs recommend Python 3.8 but 3.10 works for v0.5.0 and avoids pain with current numpy/scipy.)

### 3. MuJoCo

```bash
pip install mujoco==3.2.7
python -c "import mujoco; print('mujoco', mujoco.__version__)"
```

(Note: robosuite v1.5.1 declares `mujoco>=3.2.3`, but the latest 3.8.x rejects Sawyer/IIWA mesh inertia under stricter validation — pin to 3.2.7. The `mink` dep wants `>=3.3.6` but it's only used for the GR1 humanoid which we don't touch.)

### 4. robosuite v1.5.1

```bash
cd ~/Desktop/projects
git clone https://github.com/ARISE-Initiative/robosuite.git
cd robosuite
git checkout v1.5.1
pip install -e .
```

Smoke test (no rendering, just dynamics):

```bash
python -c "
import robosuite as suite
env = suite.make('Stack', robots='Panda', has_renderer=False, has_offscreen_renderer=True, use_camera_obs=False, control_freq=20)
env.reset()
print('robosuite Stack env OK')
"
```

### 5. robomimic v0.5.0 (Diffusion Policy lives here)

```bash
cd ~/Desktop/projects
git clone https://github.com/ARISE-Initiative/robomimic.git
cd robomimic
pip install -e .
```

**Mac patch needed:** robomimic's `setup.py` lists `egl_probe` as a hard dep for Linux multi-GPU EGL device selection. It needs cmake+make and doesn't apply to Mac. Two-line patch:

1. Remove `"egl_probe>=1.0.1",` from `setup.py` `install_requires`.
2. In `robomimic/envs/env_robosuite.py`, wrap the `import egl_probe` block in `try/except ImportError: pass`.

Then `pip install -e .` succeeds.

### 6. mimicgen (for the Stack dataset)

```bash
cd ~/Desktop/projects
git clone https://github.com/NVlabs/mimicgen.git
cd mimicgen
pip install -e .
```

### 7. PyTorch

On Mac (eval / debugging only — no GPU training):

```bash
pip install torch torchvision
```

### 8. This project

```bash
cd ~/Desktop/projects/shadow
pip install -r requirements.txt
```

### 9. Final verification

```bash
python -c "import torch, robomimic, robosuite, mimicgen, mujoco; print('all imports OK')"
```

## Part 2 — Cloud GPU (Colab Pro or GCP)

Filled in at Phase 5 when we hit training. Plan: same conda recipe, swap CPU PyTorch for CUDA 12.x build, mount the dataset from a GCS bucket or HuggingFace.

## Common gotchas

- **MuJoCo render on Mac.** If `env.render()` segfaults: `export MUJOCO_GL=glfw` (interactive) or `MUJOCO_GL=egl` (offscreen).
- **robosuite version mismatch.** robomimic v0.5 expects robosuite v1.5.x. Older robosuite silently produces wrong observation shapes — you will lose hours debugging.
- **HDF5 / h5py on Apple Silicon.** If you see "library not loaded" errors: `pip uninstall h5py && pip install --no-binary=h5py h5py`.
- **Mimicgen Stack dataset format.** Demos are stored as HDF5; image observations are *not* pre-rendered. We render images on-the-fly when building the Shadow dataset (Phase 3).
