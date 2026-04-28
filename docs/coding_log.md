# RM 编码日志 (Coding Log)

> **文件用途**：把"做了什么、为什么这样做、怎么跑、下一步做什么"以最小成本记录下来。
> 这份文档是**给未来的自己 / 协作者 / agent 读**的——不是过程旁白。
> 每一轮编码完结都在末尾追加一节，**绝不删除历史条目**。
>
> 配套文件：
> * `docs/RM_design_and_roadmap.md`：方法论 / 实验设计的单一信任源。
> * `docs/experiment_log.md`：实验运行结果（与本文件一一对应）。

---

## 阶段 1 第 1 轮 — 项目骨架与最小闭环 (2026-04-28)

### 目标
打通"从 0 到第一帧"的最小闭环：**任何贡献者只要 conda env 干净，10 分钟内即可跑通端到端 self-check**。
为后续阶段（Episode/Pattern 抽取、Predictive-Surprise、Bayesian 更新、消融、Refute 实验）准备好一个
"插件式架构"：每一层都有清晰的接口、Mock 实现、单元测试。

### 范围（本轮 ✅；下一轮 ➡）

| 模块 | 状态 | 说明 |
|---|---|---|
| 项目骨架（pyproject、目录、configs、.gitignore） | ✅ | 见 `pyproject.toml` 和 `configs/` |
| Pydantic schemas（Event/Episode/Pattern/Principle + 辅助类型） | ✅ | `src/rm/memory/schemas.py` |
| LLM 客户端（OpenAI-compat + Anthropic + Mock + 重试 + Token 计数） | ✅ | `src/rm/llm/client.py` |
| Prompts v1（P1–P6 + ReAct + RM-system） | ✅ | `src/rm/llm/prompts/v1/*.txt` |
| Embedder（SentenceTransformers + OpenAI-compat + Mock） | ✅ | `src/rm/llm/embed.py` |
| Memory Store（SQLite ⊕ Qdrant，含 in-memory 模式 + retrieve()） | ✅ | `src/rm/memory/store.py` |
| Env 基类 + MockEnv + ALFWorld 懒加载包装 | ✅ | `src/rm/envs/{base,mock_env,alfworld_env}.py` |
| Agent 基类 + RandomAgent + ReActAgent | ✅ | `src/rm/agent/{base,react}.py` |
| 工具：logging、seeding、config (Hydra-lite) | ✅ | `src/rm/utils/*.py` |
| 单元测试（43 条，全部通过） | ✅ | `tests/test_*.py` |
| Scripts：99_self_check / 00_smoke_alfworld / 01_run_react_mock | ✅ | `scripts/` |
| Writer 子模块（Episode 切分、Pattern 抽取、Principle 反思） | ➡ Round 2 | `src/rm/memory/writer.py` |
| Updater（Predictive-Surprise + Bayesian） | ➡ Round 3 | `src/rm/memory/updater.py` |
| Forgetter（utility × stability） | ➡ Round 5 | `src/rm/memory/forgetter.py` |
| ReflectiveAgent（接入 memory 检索） | ➡ Round 4 | `src/rm/agent/reflective.py` |
| Eval runner / metrics / W&B 集成 | ➡ Round 4 | `src/rm/eval/*.py` |
| Reflexion / MemGPT / A-MEM / Mem0 / AWM Baselines | ➡ Round 6 | `src/rm/baselines/*.py` |
| ALFWorld-Refute 构造脚本 | ➡ Round 7 | `scripts/05_build_refute_env.py` |

### 关键架构决策（仅记录非显然的）

1. **Schema 层强制无依赖（除 pydantic）**。
   - 原因：让 schemas 可被 store / writer / retriever / 测试 / Notebook 任意 import，不引入大依赖。
   - 后果：`embedding` 字段是 `list[float]`，不是 numpy；vector 操作在 store / embed 模块里完成。

2. **存储采用 SQLite ⊕ Qdrant 双库（而非 ChromaDB / Faiss）**。
   - SQLite：所有结构化字段（含 JSON-encoded list/dict）、ACID、零依赖。
   - Qdrant：embedding 向量 + ANN；`url=None` 时自动启用内存模式（`QdrantClient(":memory:")`），
     unit tests 与 Windows 用户**无需 docker** 即可跑通整套检索。
   - Embedding 不存 SQLite，避免行变胖；以 `item_id` 在两库间引用一致。
   - Qdrant 的点 ID 必须是 int 或 UUID，因此对任意 `item_id` 用 `uuid5(NAMESPACE_OID, item_id)` 做幂等映射。

3. **LLM 客户端只暴露三个方法**：`chat`、`chat_json`、`count_tokens`。
   - 不引入 LangChain / LlamaIndex 等抽象框架（roadmap §5.1 明确要求）。
   - JSON-mode 自带 fence / brace 修复 + 失败时让 LLM 二次自修，三层兜底。
   - 重试用 `tenacity` 指数退避；usage tracker 累计 prompt/completion/total tokens，方便后续 cost 报表。

4. **MockLLMClient + MockEmbedder + MockEnv 是一等公民**，不只是测试 fixture。
   - 让 self-check / 单测 / 离线开发完全无网络、无 GPU、无 docker。
   - MockEmbedder 用 SHA256 链式扩展（不是 blake2b，因为后者 `digest_size ≤ 64 字节`，无法支持高维向量）。

5. **配置走 OmegaConf + 简易 Hydra emulator**。
   - 不直接依赖 Hydra 的 `@hydra.main` 装饰器，避免脚本入参格式被锁死；
   - 但兼容其 `defaults` 列表与 `${oc.env:...}` interpolation；
   - **手动注册了 `now` resolver**（OmegaConf 不内置，Hydra 才有），见 `rm/utils/config.py`。

6. **Env 全部懒加载**：`build_env({"name": "alfworld"})` 才会触发 `import alfworld`。
   Windows 上没有 textworld 也能 `pip install -e ".[dev]"` 通过；只有跑 `00_smoke_alfworld.py` 才会撞墙。

### 调试中暴露并修复的 bug（保留以防复现）

| # | 现象 | 原因 | 修复 |
|---|---|---|---|
| 1 | MockEmbedder dim=64 时 `ValueError: digest_size must be between 1 and 64 bytes` | `blake2b` 上限 64 B，dim*2=128 越界 | 改用多次 SHA256 链式扩展（`sha256(chunk_idx + text)`），任意 dim 都行 |
| 2 | `OmegaConf.errors.UnsupportedInterpolationType: Unsupported interpolation type now` | OmegaConf 不内置 `now`，Hydra 才有 | 在 `rm/utils/config.py` 注册 `now` resolver |
| 3 | `_resolve_defaults` 跳过所有 group 配置（exp/llm/env/agent 均为空） | 把 `DictConfig` 当成 `dict`，`isinstance(item, dict)` 为 False | 显式 `to_container(d, resolve=False)` 转回 dict 后再判 |
| 4 | `${exp.name}` interpolation 解析失败 | `out[group] = sub` 不重新挂接父节点 | 改用 `OmegaConf.merge(out, OmegaConf.create({group: sub}))` |
| 5 | `${oc.env:HOME,~}` 报 `GrammarParseError: token recognition error at: '~'` | `~` 字面量在 `oc.env` 默认值位上不合法 | `data_path: null` + 由 `ALFWorldEnv` 在 runtime 解析 `$ALFWORLD_DATA` 或 `~/.alfworld` |

### 仓库结构（最终态）

```
.
├── pyproject.toml               # 依赖 + 工具配置（pytest/ruff/mypy）
├── README.md
├── .gitignore
├── configs/
│   ├── base.yaml
│   ├── env/   {alfworld, mock}.yaml
│   ├── llm/   {qwen7b, qwen14b, gpt4o, deepseek}.yaml
│   ├── agent/ {react, noMem, rm}.yaml
│   └── exp/   {smoke, main}.yaml
├── src/rm/
│   ├── memory/{__init__, schemas, store}.py
│   ├── llm/{__init__, client, embed, prompts}.py
│   │   └── prompts/v1/{P1_segment,P2_pattern,P3_principle,P4_predict,
│   │                    P5_judge,P6_revise,react_system,react_step,rm_system}.txt
│   ├── envs/{__init__, base, mock_env, alfworld_env}.py
│   ├── agent/{__init__, base, react}.py
│   └── utils/{__init__, logging, seeding, config}.py
├── tests/
│   ├── conftest.py
│   ├── test_schemas.py    # 9
│   ├── test_embed.py      # 5
│   ├── test_llm_mock.py   # 9
│   ├── test_store.py      # 9
│   ├── test_prompts.py    # 3
│   ├── test_agent_react.py # 5
│   └── test_config.py     # 3
├── scripts/
│   ├── 99_self_check.py
│   ├── 00_smoke_alfworld.py
│   └── 01_run_react_mock.py
├── docs/
│   ├── RM_design_and_roadmap.md
│   ├── coding_log.md
│   ├── experiment_log.md
│   └── adr/                  # 待写：ADR-001 ~ ADR-00X
├── data/{alfworld_refute, traces}/
└── notebooks/
```

### 模块依赖图（验证后）

```
        agent.react ─────► llm.client (+ prompts)
              │                  │
              │                  ▼
        envs.{base,mock,alf} ◄── utils.logging
              │                  ▲
              │                  │
        memory.store ──────► memory.schemas
              │                  ▲
              └──────── llm.embed (生产 vector)
        utils.config ───────► (omegaconf)
```

不变量：`memory.*` 只依赖 `llm.client + schemas + utils`；不依赖 `envs`/`agent`。✅

### 复现命令（拷即可用）

```bash
# 一次性安装
conda activate rm
pip install -e ".[dev]"

# 端到端 self-check（应输出 7 passed, 0 failed）
python scripts/99_self_check.py

# 单元测试（应输出 43 passed）
pytest -q

# Lint
ruff check src/ tests/ scripts/

# Mock 跑 5 任务（应 100% SR，每任务 1 步）
python scripts/01_run_react_mock.py --llm mock --n_tasks 5

# 真实 LLM 跑 Mock 环境（需先把 vLLM/qwen7b 起在 localhost:8000）
python scripts/01_run_react_mock.py --llm qwen7b --n_tasks 20

# ALFWorld 烟测（需要 [envs] extras 与 alfworld-download，**Linux/WSL** 推荐）
pip install -e ".[envs]"
alfworld-download
python scripts/00_smoke_alfworld.py --n_tasks 5
```

### 第 1 轮验收指标（实测）

| 指标 | 目标 | 实测 | 状态 |
|---|---|---|---|
| 单元测试通过数 | ≥ 30 | 43 | ✅ |
| 单元测试耗时 | < 5 s | ~1.2 s | ✅ |
| Self-check 通过 | 7/7 | 7/7 | ✅ |
| Mock SR | 100 %（5/5） | 100 %（5/5） | ✅ |
| ruff lint clean | 0 error | 0 error | ✅ |
| 装包冷启动耗时 | < 5 min | ~2 min | ✅ |

---

## 第 2 轮蓝图（下一轮要做）

> 以下条目顺序按推荐编码顺序排列；每条都对应 roadmap 的某一节，可独立交付。

1. **`memory/writer.py` — 自下而上抽象**
   - `LLM_segment_episodes(events) -> list[Episode]`，调用 P1。
   - `LLM_induce_pattern(episode_cluster) -> Pattern`，调用 P2。
   - 聚类策略：默认 sklearn KMeans + silhouette；可选 HDBSCAN（`pip install ".[cluster]"`）。
   - 写入前去重：嵌入 cosine > 0.9 触发 merge 决策。
   - **测试目标**：在 N 条人造 traj（mock LLM 返回固定 JSON）上能产出 ≥ 1 条 Pattern。

2. **`memory/updater.py` — Predictive-Surprise + Bayesian**
   - `compute_surprise(pattern, episode) -> SurpriseSignal`，三种 backend（llm_judge / embed_delta / logprob）。
   - `bayesian_update(pattern, signal) -> UpdateRecord`，写入 store 并按 `tau_low/high/rewrite_thresh` 阈值化。
   - `revise_pattern(pattern, refute_episodes) -> Pattern[]`，调用 P6，支持 refine/split。

3. **`agent/reflective.py` — RM-aware Agent**
   - 继承 ReActAgent；override `_memory_block(traj)` 用 `MemoryStore.retrieve()` 拉 Principle/Pattern/Episode。
   - 调用 P0 `rm_system` 而非 `react_system`。
   - 写入 trajectory_end 时调用 writer 链。

4. **`eval/runner.py` + `metrics.py`**
   - `Runner(env_cfg, agent_cfg, n_tasks, n_seeds)` 多任务多种子运行；
   - 指标：SR / Steps / Tokens / |M| / Transfer / RL / BRA / SMP；
   - 输出 JSON Lines + （可选）W&B。

5. **完成 Reflexion baseline 复现**（roadmap W2 硬指标）。
   - 直接从 `noahshinn/reflexion` fork verbal reflection 模块；
   - 用 GPT-3.5 在 ALFWorld unseen 上复刻论文数 ±2 pp。
   - 必须在动手做 RM eval 之前过这一关，否则后续 baseline 不可信。

### 待办风险（来自 roadmap §5.5）

- **R2 Pattern 抽取质量差**：W4 起准备 50 条 Pattern 抽取的人工抽检，定期跑 `scripts/inspect_patterns.py`。
- **R3 Surprise 度量不可靠**：A7/A8 消融的实现要在 W5 与 baseline 一起准备。
- **R6 与并发工作撞车**：每周一查 arXiv（`agent memory`、`self-evolving`、`predictive memory`）。

---

<!-- 之后每完成一轮，在此处追加一节，命名为 "## 阶段 X 第 Y 轮 — <主题> (YYYY-MM-DD)" -->
