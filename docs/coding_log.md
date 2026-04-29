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

## 阶段 1 第 2 轮 — Writer / Updater / ReflectiveAgent / Eval (2026-04-29)

### 目标
把 Round 1 的"插件骨架"灌入"四层认知机制"：
1. 自下而上的写入流（Event → Episode → Pattern → Principle，见 §2.3）；
2. 横向反馈的更新流（Predictive-Surprise + Bayesian + 自动 Revision，见 §2.5）；
3. 自上而下的检索流（query 文本 → Principle/Pattern/Episode 三层文本块，见 §2.4）；
4. RM-aware Agent 把以上都接进 ReAct loop；
5. eval runner / metrics 让结果可比较、可复现。

### 范围（本轮 ✅；下一轮 ➡）

| 模块 | 状态 | 说明 |
|---|---|---|
| `memory/writer.py` — Episode/Pattern/Principle 三阶段抽象 | ✅ | 含 KMeans + silhouette 聚类、Qdrant 向量去重合并 |
| `memory/updater.py` — Surprise + Bayesian + Revision | ✅ | 三种 surprise backend；自动触发 P6 重写 |
| `memory/retriever.py` — query → MemoryContext → prompt 文本 | ✅ | 包装 store.retrieve；含层级 token 预算 |
| `agent/reflective.py` — RM-aware ReAct | ✅ | 注入 memory_block；trajectory 结束触发 writer + updater |
| `eval/metrics.py` — SR / Steps / Tokens / `|M|` + bootstrap CI | ✅ | 不依赖 scipy，纯 Python 重采样 |
| `eval/runner.py` — 多种子 × 多任务 × 多 trial 驱动器 | ✅ | 输出 trajectories.jsonl + metrics.json + config.json |
| `scripts/02_run_rm_mock.py` — RM 端到端 mock 烟测 | ✅ | 8/8 任务，1 Pattern + 2 Principle + 7 update 全链路通 |
| 单元测试（+38 用例 = 81 总数） | ✅ | writer / updater / retriever / reflective / metrics+runner |
| `memory/forgetter.py` — utility × stability 遗忘 | ➡ Round 3 | roadmap W7 |
| Reflexion baseline 复现（W2 硬指标） | ➡ Round 3 | 必过；不过则后续 RM 数据不可信 |
| 接入真 LLM (Qwen / DeepSeek / GPT-4o) 的端到端冒烟 | ➡ Round 3 | 用 `02_run_rm_mock.py --llm <name>` 的 LLM 切换 |
| ALFWorld baseline 接线 | ➡ Round 3 | Linux/WSL 下；scripts/03_run_react_alfworld.py |

### 关键架构决策（仅记录非显然的）

1. **三种 Surprise 后端（§2.5.1 三选项）全部内置，可在 config 切换**。
   - `llm_judge`（默认）：P4 预测 → P5 1–5 分歧度 → 归一到 [0,1]。语义敏感但贵。
   - `embed_delta`：cosine_distance(embed(expected_effect), embed(actual_summary))。便宜且足够 robust，**消融用**（A7）。
   - `logprob`：未实现；占位符自动 fallback 到 `embed_delta` 并打日志。Round 3 视情况补。
   - 设计在 `MemoryUpdater(backend=...)` 一处切，方便 §4.4 A7/A8 消融。

2. **Beta 更新有"软更新带"**。
   - `s < tau_low`：α += 1（强支持）；
   - `s > tau_high`：β += 1（强反驳）；
   - 中间带：α += 1−s，β += s（按 surprise 比例软更新）。
   - 同时给 α/β 加 `alpha_max=200, beta_max=200` 的硬上限，对应 §3.3 (A3) 防止"无限累积"。

3. **Pattern revision 三选一：refine / split / discard（§2.5.3）**。
   - LLM 通过 P6 输出 decision；新生成 Pattern 设 `parent_pattern_id`、`version+=1`、α/β 重置。
   - `discard`：旧 Pattern 不删，而是把 β 加倍 + 1 → 后续检索置信度急剧下降，相当于"软淘汰"。
   - 这样保留 audit trail（原 Pattern 仍能查），符合 roadmap §2.6 "不丢信息，只是降权 + 浓缩"的定调。

4. **聚类专门处理"重复嵌入"边界**。
   - 现实：写入流第一次跑时所有 Episode 文本相似度极高，KMeans 会触发 ConvergenceWarning。
   - 处理：捕获 ConvergenceWarning 静默；当 `len(set(labels)) < 2` 时跳过 silhouette；最终保底返回 `[0]*n`，让单一大簇仍然能用作 Pattern induction 的输入。

5. **Pattern 去重用 Qdrant ANN，而不是遍历内存**。
   - 起初实现用 `store.all_patterns()` 遍历再算 cosine — 这暴露了一个**根本性架构问题**：SQLite 不存 embedding（只有 Qdrant 存），所以 `all_patterns()` 返回的 Pattern 全部 `embedding=None`，cosine 永远 0。
   - 现修复：`PatternInducer.find_near_duplicate` 调 `store.query_vectors(MemoryLayer.PATTERN, candidate.embedding, top_k=1)`，直接用 Qdrant 的 cosine score 比较 `merge_cosine` 阈值。
   - 同样的根因影响 `EpisodeClusterer.cluster(recent)`：从 SQLite 拉的 episodes 没向量，全被过滤。修复：在 writer 里加 `store.fetch_vectors(MemoryLayer.EPISODE, ep_ids)` 主动从 Qdrant 拉向量再注入。

6. **`MemoryStore.fetch_vectors(layer, ids)` 是新引入的对外 API**。
   - 用例：writer 聚类、未来 forgetter 计算几何分布。
   - 实现：用 `qdrant.retrieve(ids=..., with_vectors=True)`。失败时返回空 dict（log 警告），让上层自然降级。

7. **ReflectiveAgent 把 memory 接入 prompt 的"零侵入式"做法**。
   - 直接 override 父类 `_memory_block(traj)` 这个钩子（ReActAgent 已经在 prompt 模板里 `{memory_block}` 留了占位）；
   - on_episode_start 缓存任务文本；act() 时 `query = task | last_obs`，避免高维特征工程；
   - `retrieve()` 失败（向量维度不匹配 / Qdrant 异常）时返回空 ctx，agent 不崩。
   - 结果：把"普通 ReAct"和"RM-aware"切换只需替换 agent 实例，prompt v1 不动。

8. **eval runner 的 trajectory 序列化策略**。
   - 用 JSONL 一条 trajectory 一行：可流式写、grep 友好、不需要全部跑完才能看结果。
   - `metrics.json` 是聚合视图，独立持久化，方便对比多次运行。
   - 不依赖 W&B（设 `log_to_wandb=False` 默认）；启用时 W&B 失败也不阻塞实验。

### 调试中暴露并修复的 bug（保留以防复现）

| # | 现象 | 原因 | 修复 |
|---|---|---|---|
| 1 | `EpisodeClusterer.cluster(recent)` 始终返回 `[]`，Pattern 永远不被抽取 | SQLite 读出的 Episode 没有 `embedding` 字段，cluster 全过滤 | 写入流里加一步：`store.fetch_vectors(LAYER.EPISODE, ids)` 从 Qdrant 注入向量再聚类 |
| 2 | `PatternInducer.find_near_duplicate` 永远返回 None，无法去重 | 同 #1 — `store.all_patterns()` 不带向量 | 改用 `store.query_vectors(LAYER.PATTERN, candidate.embedding, top_k=1)` 走 Qdrant ANN |
| 3 | `test_surprise_llm_judge` 拿到 0.5（中性）而非 1.0（强反驳） | rule pattern `r"1=nearly identical"` 不匹配 P5 模板里的 `"1 = nearly identical"`（带空格） | rule 改用 P5 独有短语 `r"diverges from the prediction"` |
| 4 | KMeans 在重复 embedding 上报 ConvergenceWarning | 同一 embedding 的多个点导致 KMeans 收敛到 1 簇 | `warnings.simplefilter("ignore", ConvergenceWarning)` + 对 `set(labels) < 2` 跳过 silhouette；保底返回单簇 labels |

### 第 2 轮验收指标（实测）

| 指标 | 目标 | 实测 | 状态 |
|---|---|---|---|
| 单元测试通过数 | ≥ 70 (=43 + ~30 新) | 81 | ✅ |
| 单元测试耗时 | < 10 s | ~7 s | ✅ |
| ruff lint clean | 0 error | 0 error | ✅ |
| RM mock 端到端 SR | 100 % (8/8) | 100 % (8/8) | ✅ |
| RM mock 跑出 ≥ 1 Pattern | ≥ 1 | 1（被反复 merge） | ✅ |
| RM mock 跑出 ≥ 1 Principle | ≥ 1 | 2（每 4 traj 反思一次） | ✅ |
| RM mock 触发 ≥ 1 Bayesian update | ≥ 1 | 7 | ✅ |

### 新增/修改文件清单

```
新增：
  src/rm/memory/writer.py            (~480 行)  — Episode/Pattern/Principle 三阶段
  src/rm/memory/updater.py           (~330 行)  — Surprise + Bayesian + Revision
  src/rm/memory/retriever.py         (~140 行)  — 文本查询 → MemoryContext → prompt 文本
  src/rm/agent/reflective.py         (~170 行)  — ReActAgent + RM
  src/rm/eval/__init__.py            (~5 行)
  src/rm/eval/metrics.py             (~110 行)  — EvalMetrics + bootstrap CI
  src/rm/eval/runner.py              (~170 行)  — 多种子 × 多任务 × 多 trial
  scripts/02_run_rm_mock.py          (~150 行)  — RM 端到端 mock 烟测
  tests/test_writer.py               (~210 行 / 9 用例)
  tests/test_updater.py              (~210 行 / 13 用例)
  tests/test_retriever.py            (~70 行 / 5 用例)
  tests/test_reflective_agent.py     (~110 行 / 3 用例)
  tests/test_metrics_and_runner.py   (~90 行 / 8 用例)

修改：
  src/rm/memory/__init__.py          — 导出新写入器/检索器/更新器
  src/rm/memory/store.py             — 新增 fetch_vectors(layer, item_ids)
  src/rm/agent/__init__.py           — 导出 ReflectiveAgent
```

### 复现命令（拷即可用）

```bash
conda activate rm

# 单元测试 → 期望 "81 passed"
pytest -q

# Lint
ruff check src/ tests/ scripts/

# self-check（仍然 7/7）
python scripts/99_self_check.py

# RM 端到端 mock 烟测 → 期望 SR=8/8, |M| 中 patterns≥1 principles≥1 updates≥1
python scripts/02_run_rm_mock.py --n_tasks 8 --max_steps 5

# 切换 surprise backend 做消融对比
python scripts/02_run_rm_mock.py --n_tasks 8 --surprise_backend embed_delta
python scripts/02_run_rm_mock.py --n_tasks 8 --surprise_backend llm_judge
```

---

## 第 3 轮蓝图（下一轮要做）

> 本轮以"对外可比较的实验数字"为主，不再造新模块。

1. **Reflexion baseline 复现**（roadmap W2 硬指标，最高优先级）。
   - 路径：`src/rm/baselines/reflexion.py`，从 `noahshinn/reflexion` 抽 verbal-reflection 逻辑；
   - 适配我们的 LLMClient 与 envs 接口；
   - **验收**：在 ALFWorld eval_out_of_distribution（134 任务）上用 GPT-3.5 跑出 SR ∈ [78, 85]%（论文报）；用 Qwen2.5-7B 跑同设置（不要求达到论文数）。
   - 不过这一关，后续 RM 数据全部不可信。

2. **真 LLM 端到端 RM 跑**。
   - `02_run_rm_mock.py --llm qwen7b` / `--llm deepseek`，先在 MockEnv 验证 prompt 在真 LLM 下能产 1 个 Pattern；
   - 然后接 ALFWorld（需 WSL）跑 5–10 任务，记 traj + memory dump 到 `data/traces/`。

3. **Forgetter（roadmap W7 内容前置）**。
   - `memory/forgetter.py`：utility × stability 双轴评分 + consolidate before delete；
   - 测试：在写入 50+ patterns 后跑一次 forget，检查低分条目被合并/归档。

4. **Eval CLI**。
   - `scripts/run_eval.py --config configs/exp/main.yaml`：把 Round 2 的 Runner 接到 Hydra 配置上，一行命令出主表行。
   - 输出格式直接对齐 §4.3 主表的列。

5. **W&B 集成实测**。
   - 跑一次开 `log_to_wandb=True`，确认 metrics + trajectory hash 都进 W&B；
   - 在 `coding_log.md` 留 W&B board 链接（runs/main 板）。

### 待办风险更新

- **R2 Pattern 抽取质量**：mock 跑下抽出来的 Pattern 看着像样，但**真 LLM 第一次跑必定有大量 prompt drift**——Round 3 需要立刻准备 50 条人工抽检表，对 P1/P2/P3 各做一次 prompt 微调。
- **R5 benchmark 分歧**：在 Reflexion 复现失败时，先 freeze prompts/v1，用 GPT-4o-as-judge 反向校验数字一致性。

