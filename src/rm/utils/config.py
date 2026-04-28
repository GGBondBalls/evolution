"""Hydra/OmegaConf config loading — used by scripts and the eval runner."""

from __future__ import annotations

import datetime as _dt
from pathlib import Path
from typing import Any

from omegaconf import DictConfig, OmegaConf

PROJECT_ROOT = Path(__file__).resolve().parents[3]
CONFIGS_DIR = PROJECT_ROOT / "configs"


# OmegaConf doesn't ship a ``now`` resolver (Hydra does). Register ours so
# that ``${now:%Y%m%d_%H%M%S}`` works without pulling Hydra into every test.
def _register_resolvers() -> None:
    if not OmegaConf.has_resolver("now"):
        OmegaConf.register_new_resolver(
            "now",
            lambda fmt="%Y%m%d_%H%M%S": _dt.datetime.now().strftime(fmt),
        )


_register_resolvers()


def load_config(
    config_name: str = "base",
    overrides: list[str] | None = None,
    extra: dict[str, Any] | None = None,
) -> DictConfig:
    """Load ``configs/<config_name>.yaml`` and resolve its ``defaults`` block.

    This is a deliberately minimal Hydra emulation. It supports the subset of
    Hydra we actually need: ``defaults``-style imports of group configs and
    dotted overrides like ``llm=gpt4o`` or ``rm.surprise.tau_low=0.15``.

    For full Hydra features, run a script wrapped in ``@hydra.main`` instead.
    """
    base_path = CONFIGS_DIR / f"{config_name}.yaml"
    if not base_path.exists():
        raise FileNotFoundError(f"Config not found: {base_path}")

    cfg = OmegaConf.load(base_path)
    cfg = _resolve_defaults(cfg)

    if overrides:
        for ov in overrides:
            if "=" not in ov:
                raise ValueError(f"Bad override (need 'key=value'): {ov}")
            key, value = ov.split("=", 1)
            if "/" in key or key in {"llm", "env", "agent", "exp"}:
                # Group override — load the group config and merge under that key.
                key = key.replace("/", "")
                group_path = CONFIGS_DIR / key / f"{value}.yaml"
                if not group_path.exists():
                    raise FileNotFoundError(group_path)
                cfg[key] = OmegaConf.load(group_path)
            else:
                OmegaConf.update(cfg, key, _coerce(value), merge=True)

    if extra:
        cfg = OmegaConf.merge(cfg, OmegaConf.create(extra))

    OmegaConf.resolve(cfg)
    return cfg  # type: ignore[return-value]


def _resolve_defaults(cfg: DictConfig) -> DictConfig:
    """Walk the ``defaults`` list and merge group configs onto the base."""
    raw_defaults = list(cfg.pop("defaults", [])) if "defaults" in cfg else []
    # Normalise list elements: each is either the string "_self_" or a
    # one-key mapping ``{group: name}``. OmegaConf returns DictConfig, not
    # dict, so we coerce explicitly.
    items: list[str | dict[str, Any]] = []
    for d in raw_defaults:
        if isinstance(d, str):
            items.append(d)
        else:
            items.append(dict(OmegaConf.to_container(d, resolve=False)))  # type: ignore[arg-type]

    out: DictConfig = OmegaConf.create({})  # type: ignore[assignment]
    has_self = "_self_" in items
    for item in items:
        if item == "_self_":
            out = OmegaConf.merge(out, cfg)  # type: ignore[assignment]
            continue
        assert isinstance(item, dict)
        for group, name in item.items():
            if name is None:
                continue
            group_path = CONFIGS_DIR / group / f"{name}.yaml"
            if not group_path.exists():
                raise FileNotFoundError(group_path)
            sub = OmegaConf.load(group_path)
            # Use merge under a wrapper so the new sub-config gets re-parented
            # — this matters for cross-group interpolations like ``${exp.name}``.
            out = OmegaConf.merge(out, OmegaConf.create({group: sub}))  # type: ignore[assignment]
    if not has_self:
        out = OmegaConf.merge(out, cfg)  # type: ignore[assignment]
    return out


def _coerce(v: str) -> Any:
    """Best-effort string→primitive cast for overrides."""
    if v.lower() in {"true", "false"}:
        return v.lower() == "true"
    if v.lower() in {"null", "none"}:
        return None
    try:
        return int(v)
    except ValueError:
        pass
    try:
        return float(v)
    except ValueError:
        pass
    return v
