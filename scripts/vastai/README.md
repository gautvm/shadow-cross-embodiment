# Vast.ai training pipeline

End-to-end recipe to train a Shadow variant on a rented Vast.ai GPU with
checkpoint exfiltration to GitHub releases (so we don't lose work if the
instance dies).

## One-time setup

1. **Vast.ai account + funded balance.** Load $20 via web (https://cloud.vast.ai/billing/).
2. **SSH key uploaded** to https://cloud.vast.ai/account/ → Keys.
3. **GitHub PAT** with `repo` write scope. Create at
   https://github.com/settings/tokens?scopes=repo. Save it as env var:

   ```bash
   export GH_TOKEN=ghp_xxxxx
   ```

4. **GitHub release `v0-checkpoints` exists.** One-time:

   ```bash
   gh release create v0-checkpoints -R gautvm/shadow-cross-embodiment \
     --title "Trained checkpoints" --notes "Diffusion policy checkpoints from cloud training"
   ```

## Per-variant launch

```bash
# 1. Find a cheap GPU
python scripts/vastai/launch.py search
# → table of options. Pick the leftmost ID of an RTX 3090 or 4090 you like.

# 2. Launch training (token must be in env)
python scripts/vastai/launch.py launch shadow_noise <OFFER_ID>
# → prints INSTANCE_ID and the ssh-attach command

# 3. Watch setup + training
vastai ssh-url <INSTANCE_ID>
ssh <addr> -t 'tmux attach -t train'

# 4. Destroy when done (manual — cost stops here)
python scripts/vastai/launch.py destroy <INSTANCE_ID>
```

## What happens on the instance

When the instance boots, `--onstart` runs a hook that:
1. Installs git/tmux/wget
2. Clones `gautvm/shadow-cross-embodiment` to `/workspace/shadow`
3. Starts `scripts/vastai/run.sh` inside a tmux session named `train`

`run.sh` then:
1. Installs gh CLI + authenticates with `$GH_TOKEN`
2. Runs `setup.sh` if not done (clones robosuite/robomimic/mimicgen, applies
   patches, pins PyTorch 2.4.1 cu121, removes flax)
3. Downloads the dataset for the requested variant from the v0-data release
4. Checks the v0-checkpoints release for an existing `<variant>_seed<N>.pth`
   and resumes from it if found (otherwise trains from scratch)
5. Spawns a background watcher that pushes `last.pth` to the v0-checkpoints
   release every 60 seconds (whenever it's been updated)
6. Runs `train.py`
7. On completion, pushes the final `last.pth` to the release

## Recovery from instance death

Just `launch` again with the same variant. The script auto-resumes from the
last checkpoint in the v0-checkpoints release. Worst-case loss: ~10 epochs
(robomimic saves `last.pth` every 10 epochs per `configs/base.json`, and the
watcher pushes within 60s of that).

## Cost expectations

- RTX 3090 spot: ~$0.20-0.30/hr × ~3h per variant = **~$1 per variant**
- 3 variants run sequentially: ~$3-4
- 3 variants run in parallel (3 instances): same total cost, ~3h wall clock

## Cleanup checklist after all training done

```bash
# 1. Confirm all checkpoints uploaded
gh release view v0-checkpoints -R gautvm/shadow-cross-embodiment

# 2. Destroy ALL instances
python scripts/vastai/launch.py status                # list any survivors
python scripts/vastai/launch.py destroy <ID>          # repeat per instance

# 3. (Optional) remove credit card from Vast.ai billing page

# 4. Rotate the GH_TOKEN if it was exposed in shell history
```
