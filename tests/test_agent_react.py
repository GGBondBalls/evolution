"""ReAct agent end-to-end on MockEnv with MockLLMClient."""

from __future__ import annotations

from rm.agent.react import ReActAgent, _parse_thought_action
from rm.envs.mock_env import MockEnv
from rm.llm.client import MockLLMClient


def test_parse_thought_action_basic():
    t, a = _parse_thought_action("Thought: I should look.\nAction: look")
    assert t.startswith("I should") and a == "look"


def test_parse_thought_action_with_extras():
    t, a = _parse_thought_action(
        "Some prefix\nThought: think.\nAction: say GOAL\ntrailing text"
    )
    assert t == "think." and a.startswith("say GOAL")


def test_parse_falls_back_to_first_nonempty_line():
    t, a = _parse_thought_action("look\n", fallback=["wait"])
    assert a == "look"


def test_react_solves_mock_env_with_scripted_llm():
    llm = MockLLMClient(default="Thought: speak the magic word.\nAction: say GOAL")
    agent = ReActAgent(llm=llm, max_steps=10)
    env = MockEnv(seed=1, horizon=5, goal_keyword="GOAL")
    traj = agent.run_episode(env, task_idx=0)
    assert traj.success
    assert traj.n_steps == 1
    assert traj.steps[0].action.startswith("say GOAL")


def test_react_runs_out_of_horizon_when_wrong_action():
    llm = MockLLMClient(default="Thought: idle.\nAction: wait")
    agent = ReActAgent(llm=llm, max_steps=10)
    env = MockEnv(seed=2, horizon=3, goal_keyword="GOAL")
    traj = agent.run_episode(env, task_idx=0)
    assert not traj.success
    assert traj.n_steps == 3
