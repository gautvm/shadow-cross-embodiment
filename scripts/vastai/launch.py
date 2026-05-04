"""Vast.ai launcher for Shadow training. Run from local Mac.

Three subcommands:
  search                   list cheap available GPUs that meet our requirements
  launch <variant> <id>    rent the offer at <id> and kick off training
  destroy <instance_id>    destroy a running instance (cleanup)

Auth: requires `vastai set api-key <key>` already done.
Auth: requires GH_TOKEN env var (GitHub PAT with write access to the release repo).

Examples:
  python launch.py search
  python launch.py launch shadow_noise 12345678
  python launch.py destroy 99887766
"""
from __future__ import annotations

import argparse
import ast
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

VASTAI = str(Path.home() / "Library/Python/3.11/bin/vastai")
if not Path(VASTAI).exists():
    VASTAI = "vastai"

REPO = "gautvm/shadow-cross-embodiment"
DOCKER_IMAGE = "pytorch/pytorch:2.4.1-cuda12.1-cudnn9-runtime"

VALID_VARIANTS = {"rgb", "shadow_vanilla", "shadow_noise"}


def _run(cmd: list[str], **kw) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, check=True, text=True, capture_output=True, **kw)


def cmd_search(args):
    """Print cheap, reliable GPUs that fit our workload."""
    query = (
        f"gpu_name in [RTX_3090,RTX_4090,RTX_3080,RTX_4080] "
        f"reliability>0.97 "
        f"num_gpus=1 "
        f"dph<{args.max_dph} "
        f"disk_space>40 "
        f"cuda_vers>=12.1 "
        f"rentable=true"
    )
    out = _run([VASTAI, "search", "offers", query, "-o", "dph+"])
    print(out.stdout[:3000])
    print()
    print("Pick an ID from the leftmost column. Then:")
    print(f"  python {sys.argv[0]} launch <variant> <ID>")


def cmd_launch(args):
    if args.variant not in VALID_VARIANTS:
        print(f"variant must be one of {VALID_VARIANTS}")
        sys.exit(1)

    gh_token = os.environ.get("GH_TOKEN")
    if not gh_token:
        print("error: GH_TOKEN env var required (GitHub PAT with repo write access)")
        print("  create at https://github.com/settings/tokens?scopes=repo")
        sys.exit(1)

    print(f"[launch] renting offer {args.offer_id} for variant={args.variant}")

    # The on-start script runs once when the instance boots. It clones the
    # shadow repo, then exec's run.sh in a tmux session named 'train' so we
    # can ssh in later and `tmux attach` to watch.
    onstart = f"""\
set -e
cd /workspace
apt-get update -qq && apt-get install -y -qq git tmux wget
[ -d shadow ] || git clone -q https://github.com/{REPO}.git shadow
cd shadow && git pull -q
chmod +x scripts/vastai/*.sh
tmux new-session -d -s train \\
  "VARIANT={args.variant} SEED={args.seed} EPOCHS={args.epochs} GH_TOKEN={gh_token} \
   bash /workspace/shadow/scripts/vastai/run.sh 2>&1 | tee /workspace/train.log"
echo 'training started in tmux session: train'
"""

    onstart_path = Path("/tmp/vastai_onstart.sh")
    onstart_path.write_text(onstart)

    out = _run([
        VASTAI, "create", "instance", str(args.offer_id),
        "--image", DOCKER_IMAGE,
        "--disk", str(args.disk),
        "--ssh",
        "--onstart", str(onstart_path),
    ])
    print(out.stdout)

    # vastai prints e.g. "Started. {'success': True, 'new_contract': 36097922, ...}"
    # — Python dict syntax with single quotes, not JSON. Pull the {...} substring
    # out of the last non-empty line and parse with ast.literal_eval.
    iid = None
    for line in reversed(out.stdout.strip().splitlines()):
        m = re.search(r"\{.*\}", line)
        if not m:
            continue
        try:
            result = ast.literal_eval(m.group(0))
        except (ValueError, SyntaxError):
            try:
                result = json.loads(m.group(0))
            except json.JSONDecodeError:
                continue
        iid = result.get("new_contract") or result.get("id")
        if iid:
            break

    print()
    print("=" * 60)
    if iid:
        print(f"INSTANCE ID: {iid}")
        print()
        print("Watch setup + training (wait ~3 min for boot, then):")
        print(f"  {VASTAI} ssh-url {iid}    # get ssh address")
        print(f"  ssh <addr> -t 'tmux attach -t train'")
        print()
        print("Tail log without ssh:")
        print(f"  ssh <addr> 'tail -f /workspace/train.log'")
        print()
        print("Destroy when done (do this!):")
        print(f"  python {sys.argv[0]} destroy {iid}")
    else:
        print("Could not parse instance ID — check vastai console:")
        print("  https://cloud.vast.ai/instances/")
    print("=" * 60)


def cmd_destroy(args):
    print(f"[destroy] terminating instance {args.instance_id}")
    out = _run([VASTAI, "destroy", "instance", str(args.instance_id)])
    print(out.stdout)


def cmd_status(args):
    """List all running instances + their cost."""
    out = _run([VASTAI, "show", "instances"])
    print(out.stdout)


def main():
    p = argparse.ArgumentParser()
    sp = p.add_subparsers(dest="cmd", required=True)

    sp_search = sp.add_parser("search", help="list cheap available GPUs")
    sp_search.add_argument("--max_dph", type=float, default=0.50,
                           help="max $/hr (default 0.50)")
    sp_search.set_defaults(func=cmd_search)

    sp_launch = sp.add_parser("launch", help="rent + start training")
    sp_launch.add_argument("variant", choices=sorted(VALID_VARIANTS))
    sp_launch.add_argument("offer_id", type=int)
    sp_launch.add_argument("--seed", type=int, default=0)
    sp_launch.add_argument("--epochs", type=int, default=200)
    sp_launch.add_argument("--disk", type=int, default=50, help="disk GB")
    sp_launch.set_defaults(func=cmd_launch)

    sp_destroy = sp.add_parser("destroy", help="terminate an instance")
    sp_destroy.add_argument("instance_id", type=int)
    sp_destroy.set_defaults(func=cmd_destroy)

    sp_status = sp.add_parser("status", help="list running instances")
    sp_status.set_defaults(func=cmd_status)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
