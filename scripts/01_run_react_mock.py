"""Run ReAct on MockEnv with a real or mock LLM.

This is the lowest-cost end-to-end run. With ``--llm mock`` it needs *nothing*
external; with ``--llm <config-name>`` it loads ``configs/llm/<config-name>.yaml``
and hits the corresponding endpoint.

Examples::

    # all-mock (zero-cost smoke; expected: 100% success)
    python scripts/01_run_react_mock.py --llm mock --n_tasks 5

    # real LLM via configs/llm/qwen7b.yaml (vLLM at localhost:8000)
    python scripts/01_run_react_mock.py --llm qwen7b --n_tasks 20
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from omegaconf import OmegaConf

from rm.agent.react import ReActAgent
from rm.envs.mock_env import MockEnv
from rm.llm.client import MockLLMClient, build_client
from rm.utils.logging import get_logger, setup_logging
from rm.utils.seeding import set_seed

logger = get_logger(__name__)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--llm", default="mock", help="config name under configs/llm/, or 'mock'")
    ap.add_argument("--n_tasks", type=int, default=5)
    ap.add_argument("--horizon", type=int, default=10)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    setup_logging(level="INFO")
    set_seed(args.seed)

    if args.llm == "mock":
        llm = MockLLMClient(default="Thought: speak the magic word.\nAction: say GOAL")
    else:
        cfg_path = Path(__file__).resolve().parents[1] / "configs" / "llm" / f"{args.llm}.yaml"
        if not cfg_path.exists():
            raise FileNotFoundError(cfg_path)
        cfg = OmegaConf.load(cfg_path)
        OmegaConf.resolve(cfg)
        llm = build_client(OmegaConf.to_container(cfg, resolve=True))

    env = MockEnv(seed=args.seed, horizon=args.horizon, goal_keyword="GOAL")
    agent = ReActAgent(llm=llm, max_steps=args.horizon)

    successes = 0
    total_steps = 0
    for i in range(args.n_tasks):
        traj = agent.run_episode(env, task_idx=i)
        successes += int(traj.success)
        total_steps += traj.n_steps
        logger.info(
            f"[task {i:>3}] success={traj.success} steps={traj.n_steps:>2} "
            f"action='{traj.steps[0].action[:40] if traj.steps else ''}'"
        )

    logger.info(
        f"Summary: SR={successes}/{args.n_tasks} ({successes / args.n_tasks * 100:.1f}%), "
        f"avg_steps={total_steps / args.n_tasks:.2f}"
    )
    if hasattr(llm, "usage"):
        u = llm.usage
        logger.info(f"LLM usage: calls={u.calls}, tokens={u.total} (prompt={u.prompt}, completion={u.completion})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
