"""End-to-end sanity check — no network, no GPU, no docker.

Run::

    python scripts/99_self_check.py

It exercises:
* imports for every module we shipped this round
* MockEmbedder + MemoryStore round-trip (in-memory Qdrant)
* MockLLMClient parses thought/action
* MockEnv runs a full ReAct loop and the trajectory writes to the store

Exit code is 0 iff every check passes.
"""

from __future__ import annotations

import sys
import traceback
from pathlib import Path

# Allow running from a fresh checkout (no ``pip install -e .`` yet).
_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))


def main() -> int:
    print("=== Reflective Memory — self-check ===\n")
    ok = 0
    fail = 0

    def check(name: str, fn) -> None:
        nonlocal ok, fail
        print(f"[ ... ] {name}", end="\r")
        try:
            fn()
        except Exception:
            print(f"[FAIL ] {name}")
            traceback.print_exc()
            fail += 1
        else:
            print(f"[ OK  ] {name}")
            ok += 1

    # -------------------------------------------------------------- #
    # 1. Module imports                                               #
    # -------------------------------------------------------------- #

    def import_all():
        # Touch every module shipped this round to ensure they all import.
        import importlib
        for name in [
            "rm",
            "rm.memory.schemas", "rm.memory.store", "rm.memory",
            "rm.llm.client", "rm.llm.embed", "rm.llm.prompts", "rm.llm",
            "rm.envs.base", "rm.envs.mock_env", "rm.envs",
            "rm.agent.base", "rm.agent.react", "rm.agent",
            "rm.utils.logging", "rm.utils.seeding", "rm.utils.config", "rm.utils",
        ]:
            importlib.import_module(name)
        from rm import __version__
        from rm.memory.schemas import MemoryLayer
        assert __version__
        assert MemoryLayer.EVENT.value == "event"

    check("import all rm modules", import_all)

    # -------------------------------------------------------------- #
    # 2. Pydantic schemas                                             #
    # -------------------------------------------------------------- #

    def schemas_basic():
        from rm.memory.schemas import Pattern

        p = Pattern(condition="c", action_template="a", expected_effect="e",
                    alpha=4, beta=2)
        assert abs(p.confidence - 4 / 6) < 1e-6

    check("Pattern.confidence", schemas_basic)

    # -------------------------------------------------------------- #
    # 3. Embedder                                                     #
    # -------------------------------------------------------------- #

    def embedder_basic():
        from rm.llm.embed import MockEmbedder

        e = MockEmbedder(dim=32)
        v = e.encode_one("hello")
        assert len(v) == 32
        assert e.encode_one("hello") == v

    check("MockEmbedder deterministic", embedder_basic)

    # -------------------------------------------------------------- #
    # 4. Store + retrieval                                            #
    # -------------------------------------------------------------- #

    def store_roundtrip():
        from rm.llm.embed import MockEmbedder
        from rm.memory.schemas import Pattern, RetrievalQuery
        from rm.memory.store import MemoryStore

        emb = MockEmbedder(dim=32)
        store = MemoryStore(sqlite_path=":memory:", qdrant_url=None,
                             collection_prefix="rmcheck", vector_size=32)
        for cond in ["open the drawer", "microwave food", "wash dish"]:
            store.write_pattern(Pattern(condition=cond, action_template="a",
                                         expected_effect="e", alpha=4, beta=1,
                                         embedding=emb.encode_one(cond)))
        q = emb.encode_one("open drawer")
        ctx = store.retrieve(q, RetrievalQuery(query_text="open drawer", k_pattern=2,
                                                min_principle_conf=0.0,
                                                min_pattern_conf=0.0))
        assert any("open" in p.condition for p in ctx.patterns), \
            f"Expected 'open' pattern in retrieval; got {[p.condition for p in ctx.patterns]}"
        store.close()

    check("MemoryStore retrieval", store_roundtrip)

    # -------------------------------------------------------------- #
    # 5. LLM mock                                                     #
    # -------------------------------------------------------------- #

    def mock_llm_chat():
        from rm.llm.client import MockLLMClient

        c = MockLLMClient(default="Thought: t.\nAction: look")
        out = c.chat([{"role": "user", "content": "x"}])
        assert out.text.startswith("Thought")

    check("MockLLMClient chat", mock_llm_chat)

    # -------------------------------------------------------------- #
    # 6. End-to-end ReAct on MockEnv                                  #
    # -------------------------------------------------------------- #

    def react_on_mock_env():
        from rm.agent.react import ReActAgent
        from rm.envs.mock_env import MockEnv
        from rm.llm.client import MockLLMClient

        llm = MockLLMClient(default="Thought: speak.\nAction: say GOAL")
        env = MockEnv(seed=0, horizon=5, goal_keyword="GOAL")
        agent = ReActAgent(llm=llm, max_steps=5)
        traj = agent.run_episode(env, task_idx=0)
        assert traj.success and traj.n_steps == 1, \
            f"Expected 1-step success; got success={traj.success}, n_steps={traj.n_steps}"

    check("ReAct on MockEnv (1-step solve)", react_on_mock_env)

    # -------------------------------------------------------------- #
    # 7. Prompts                                                      #
    # -------------------------------------------------------------- #

    def prompts_load():
        from rm.llm.prompts import list_prompts, load_prompt

        names = list_prompts("v1")
        for required in ["P1_segment", "P2_pattern", "P4_predict",
                         "P5_judge", "P6_revise", "react_system", "react_step"]:
            assert required in names, f"Missing prompt: {required}"
        load_prompt("react_system").format(task_description="x")

    check("Prompt v1 templates load", prompts_load)

    # -------------------------------------------------------------- #
    print(f"\n=== {ok} passed, {fail} failed ===")
    return 0 if fail == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
