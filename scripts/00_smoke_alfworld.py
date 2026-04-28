"""Random-policy smoke test on ALFWorld.

Goal: prove that the env wires up correctly and we can take steps. **Not**
expected to solve any task. ALFWorld depends on textworld (Linux-only); on
Windows, run this under WSL2.

Run::

    python scripts/00_smoke_alfworld.py --n_tasks 5 --max_steps 30
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from rm.agent.base import RandomAgent
from rm.envs.alfworld_env import ALFWorldEnv
from rm.utils.logging import get_logger, setup_logging
from rm.utils.seeding import set_seed

logger = get_logger(__name__)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_tasks", type=int, default=5)
    ap.add_argument("--max_steps", type=int, default=30)
    ap.add_argument("--split", default="eval_out_of_distribution")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    setup_logging(level="INFO")
    set_seed(args.seed)

    cfg = {
        "name": "alfworld",
        "split": args.split,
        "batch_size": 1,
        "max_steps": args.max_steps,
        "type": "AlfredTWEnv",
    }
    env = ALFWorldEnv(cfg)
    agent = RandomAgent(max_steps=args.max_steps, seed=args.seed)

    successes = 0
    for i in range(args.n_tasks):
        traj = agent.run_episode(env, task_idx=i)
        successes += int(traj.success)
        logger.info(
            f"[task {i:>3}] steps={traj.n_steps:>3} reward={traj.total_reward:>5.1f} "
            f"success={traj.success}  task='{traj.task_description[:80]}'"
        )

    logger.info(f"Summary: {successes}/{args.n_tasks} succeeded by random policy.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
