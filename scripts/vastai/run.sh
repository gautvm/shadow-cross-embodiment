#!/usr/bin/env bash
# Full training pipeline for one variant on a Vast.ai instance.
# Idempotent — safe to re-run if the instance restarts.
#
# Usage: VARIANT=shadow_noise SEED=0 EPOCHS=200 GH_TOKEN=ghp_xxx ./run.sh
#
# Required env:
#   VARIANT       rgb | shadow_vanilla | shadow_noise
#   GH_TOKEN      GitHub PAT with write access to the release repo
# Optional env:
#   SEED          default 0
#   EPOCHS        default 200
#   GITHUB_REPO   default https://github.com/gautvm/shadow-cross-embodiment.git
#   RELEASE_TAG   default v0-checkpoints

set -euo pipefail

: "${VARIANT:?Set VARIANT (rgb | shadow_vanilla | shadow_noise)}"
: "${GH_TOKEN:?Set GH_TOKEN (GitHub PAT for checkpoint uploads)}"
SEED="${SEED:-0}"
EPOCHS="${EPOCHS:-200}"
GITHUB_REPO="${GITHUB_REPO:-https://github.com/gautvm/shadow-cross-embodiment.git}"
RELEASE_TAG="${RELEASE_TAG:-v0-checkpoints}"

cd /workspace

echo "[run] variant=$VARIANT seed=$SEED epochs=$EPOCHS"

# 1. Install the gh CLI (used to push checkpoints to release)
if ! command -v gh >/dev/null 2>&1; then
  echo "[run] installing gh CLI"
  type -p curl >/dev/null || (apt-get update && apt-get install -y curl)
  curl -fsSL https://cli.github.com/packages/githubcli-archive-keyring.gpg \
    | dd of=/usr/share/keyrings/githubcli-archive-keyring.gpg 2>/dev/null
  chmod go+r /usr/share/keyrings/githubcli-archive-keyring.gpg
  echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/githubcli-archive-keyring.gpg] https://cli.github.com/packages stable main" \
    > /etc/apt/sources.list.d/github-cli.list
  apt-get update -qq && apt-get install -y -qq gh
fi
# gh CLI auto-uses GH_TOKEN env var. `gh auth login --with-token` errors
# with "GH_TOKEN env var in use" when both are set, killing run.sh under
# `set -e`. Skip the explicit login — env var is sufficient.
gh auth status 2>&1 | head -3 || true

# 2. Run install + patches if not done already
[ -d robomimic ] || bash /workspace/shadow/scripts/vastai/setup.sh

# 3. Clone (or update) the shadow project
if [ ! -d shadow ]; then
  git clone -q "$GITHUB_REPO" shadow
fi
(cd shadow && git pull -q)

# 4. Download dataset for the selected variant (skip if already present)
mkdir -p shadow/datasets/core shadow/datasets/shadow
REL_DATA=https://github.com/gautvm/shadow-cross-embodiment/releases/download/v0-data
case "$VARIANT" in
  rgb)
    if [ ! -f shadow/datasets/core/stack_d0.hdf5 ]; then
      python3 /workspace/mimicgen/mimicgen/scripts/download_datasets.py \
        --download_dir shadow/datasets --dataset_type core --tasks stack_d0
    fi
    ;;
  shadow_vanilla)
    [ -f shadow/datasets/shadow/stack_d0_shadow.hdf5 ] || \
      wget -q --show-progress -O shadow/datasets/shadow/stack_d0_shadow.hdf5 "$REL_DATA/stack_d0_shadow.hdf5"
    ;;
  shadow_noise)
    [ -f shadow/datasets/shadow/stack_d0_shadow_noise.hdf5 ] || \
      wget -q --show-progress -O shadow/datasets/shadow/stack_d0_shadow_noise.hdf5 "$REL_DATA/stack_d0_shadow_noise.hdf5"
    ;;
esac

# 5. Resume from previously-uploaded checkpoint if it exists in the release
RESUME_ARG=""
RESUME_PATH="/workspace/shadow/checkpoints/${VARIANT}_seed${SEED}.pth"
mkdir -p /workspace/shadow/checkpoints
if gh release download "$RELEASE_TAG" -R gautvm/shadow-cross-embodiment \
    -p "${VARIANT}_seed${SEED}.pth" -D /workspace/shadow/checkpoints/ 2>/dev/null; then
  echo "[run] resuming from prior checkpoint $RESUME_PATH"
  RESUME_ARG="--resume_from $RESUME_PATH"
else
  echo "[run] no prior checkpoint found; training from scratch"
fi

# Where last.pth actually lands. robomimic's `output_dir` resolves
# relative to the installed package, so models end up under
# /workspace/robomimic/robomimic/results/runs/<exp>/<timestamp>/last.pth
# rather than in our project's results/runs. Search both, newest wins.
LAST_PTH_SEARCH=(/workspace/robomimic/robomimic/results/runs /workspace/shadow/results/runs)

find_latest_last_pth() {
  find "${LAST_PTH_SEARCH[@]}" -name 'last.pth' -printf '%T@ %p\n' 2>/dev/null \
    | sort -nr | head -1 | cut -d' ' -f2-
}

# 6. Background watcher: every 60 min, push the newest last.pth to the release.
# Per-minute uploads of the 1.2GB diffusion-policy ckpt cost ~$2/hr in Vast
# bandwidth — disastrous for an 8-hour run. Hourly uploads are sufficient
# given that disk-resident numbered ckpts (model_epoch_X.pth) provide
# finer-grained recovery.
(
  cd /workspace/shadow
  LAST_MTIME=0
  while true; do
    sleep 3600
    LAST=$(find_latest_last_pth)
    [ -z "$LAST" ] && continue
    MTIME=$(stat -c%Y "$LAST" 2>/dev/null || echo 0)
    if [ "$MTIME" -gt "$LAST_MTIME" ]; then
      cp "$LAST" "checkpoints/${VARIANT}_seed${SEED}.pth"
      gh release upload "$RELEASE_TAG" "checkpoints/${VARIANT}_seed${SEED}.pth" \
        --clobber -R gautvm/shadow-cross-embodiment 2>&1 | tail -2 || true
      LAST_MTIME=$MTIME
      echo "[watcher] pushed checkpoint at $(date +%H:%M:%S) ($LAST)"
    fi
  done
) &
WATCHER_PID=$!
trap "kill $WATCHER_PID 2>/dev/null || true" EXIT

# 7. Train
cd /workspace/shadow
python3 train.py --variant "$VARIANT" --seed "$SEED" --epochs "$EPOCHS" $RESUME_ARG

# 8. Final push (in case watcher missed the last write)
LAST=$(find_latest_last_pth)
if [ -n "$LAST" ]; then
  cp "$LAST" "checkpoints/${VARIANT}_seed${SEED}.pth"
  gh release upload "$RELEASE_TAG" "checkpoints/${VARIANT}_seed${SEED}.pth" \
    --clobber -R gautvm/shadow-cross-embodiment
  echo "[run] FINAL checkpoint uploaded to release $RELEASE_TAG"
fi

echo "[run] DONE — variant=$VARIANT seed=$SEED"
