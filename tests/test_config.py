"""Config loading."""

from __future__ import annotations

from rm.utils.config import load_config


def test_load_base_resolves_defaults():
    cfg = load_config("base")
    assert "llm" in cfg
    assert "env" in cfg
    assert "agent" in cfg
    assert "exp" in cfg
    assert cfg.seed == 42


def test_overrides_swap_groups():
    cfg = load_config("base", overrides=["env=mock"])
    assert cfg.env.name == "mock"


def test_dotted_override_coerces_int():
    cfg = load_config("base", overrides=["seed=7"])
    assert cfg.seed == 7
