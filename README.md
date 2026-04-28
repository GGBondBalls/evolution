# Reflective Memory (RM)

> **Hierarchical, self-evolving memory for LLM agents.**
> Event → Episode → Pattern → Principle, updated by predictive surprise & Bayesian belief.

This repository implements the system described in
[`docs/RM_design_and_roadmap.md`](docs/RM_design_and_roadmap.md). That document is
the single source of truth for the design; this README only covers how to get
the code running.

## Quick start

```bash
# 1. Activate the prepared conda env (Python 3.12)
conda activate rm

# 2. Install in editable mode with dev / embed / track extras (no env deps)
pip install -e ".[round1]"

# 3. Sanity-check the install
python scripts/99_self_check.py

# 4. Run unit tests
pytest -q
```

To run the agent on ALFWorld you also need the env extras
(Linux / WSL strongly recommended, see `docs/coding_log.md` for the Windows
caveat):

```bash
pip install -e ".[envs]"
alfworld-download         # download ALFWorld data
python scripts/00_smoke_alfworld.py
```

## Layout

```
src/rm/
├── memory/    # Event/Episode/Pattern/Principle store + writer/retriever/updater
├── llm/       # LLM client + embedder + versioned prompts
├── agent/     # ReAct + Reflective agents
├── envs/      # ALFWorld / ScienceWorld / WebShop wrappers (lazy import)
├── eval/      # runner / metrics / refute eval
└── utils/     # logging, seeding, config
```

See `docs/coding_log.md` for the running build journal and
`docs/experiment_log.md` for the experiment journal.
