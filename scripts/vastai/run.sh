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
echo "$GH_TOKEN" | gh auth login --with-token

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

# 6. Background watcher: every 60s, find the latest last.pth and push to release
(
  cd /workspace/shadow
  while true; do
    sleep 60
    LAST=$(find results/runs -name 'last.pth' -newer /tmp/last_pushed 2>/dev/null | head -1 || true)
    if [ -n "$LAST" ]; then
      cp "$LAST" "checkpoints/${VARIANT}_seed${SEED}.pth"
      gh release upload "$RELEASE_TAG" "checkpoints/${VARIANT}_seed${SEED}.pth" \
        --clobber -R gautvm/shadow-cross-embodiment 2>&1 | tail -2 || true
      touch /tmp/last_pushed
      echo "[watcher] pushed checkpoint at $(date +%H:%M:%S)"
    fi
  done
) &
WATCHER_PID=$!
trap "kill $WATCHER_PID 2>/dev/null || true" EXIT

# 7. Train
cd /workspace/shadow
python3 train.py --variant "$VARIANT" --seed "$SEED" --epochs "$EPOCHS" $RESUME_ARG

# 8. Final push (in case watcher missed the last write)
LAST=$(find results/runs -name 'last.pth' | head -1)
if [ -n "$LAST" ]; then
  cp "$LAST" "checkpoints/${VARIANT}_seed${SEED}.pth"
  gh release upload "$RELEASE_TAG" "checkpoints/${VARIANT}_seed${SEED}.pth" \
    --clobber -R gautvm/shadow-cross-embodiment
  echo "[run] FINAL checkpoint uploaded to release $RELEASE_TAG"
fi

echo "[run] DONE — variant=$VARIANT seed=$SEED"
