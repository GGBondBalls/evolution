"""Prompt template loading & substitution."""

from __future__ import annotations

import pytest

from rm.llm.prompts import list_prompts, load_prompt


def test_list_v1_contains_core_prompts():
    names = set(list_prompts("v1"))
    for required in {"P1_segment", "P2_pattern", "P4_predict", "P5_judge",
                     "P6_revise", "react_system", "react_step"}:
        assert required in names


def test_react_system_format():
    p = load_prompt("react_system")
    out = p.format(task_description="Pick up the apple")
    assert "Pick up the apple" in out


def test_missing_var_raises():
    p = load_prompt("react_step")
    with pytest.raises(KeyError):
        p.format(memory_block="", trajectory="", observation="o")  # missing 'admissible'
