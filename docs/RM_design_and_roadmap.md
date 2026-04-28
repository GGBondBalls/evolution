# 反思式记忆 (Reflective Memory, RM) — 选题 3 完整设计与编码路线图

> **文档定位**：这是一份"边写论文边写代码"的单一信任源 (single source of truth)。
> - 第 1–3 章 = 方法论 (论文 §3 的草稿底子)
> - 第 4 章 = 实验 (论文 §4 的草稿底子)
> - 第 5–6 章 = 工程实施 (从 Day 1 编码的依据)
> - 附录 = 可直接复用的 Schema 与 Prompt 模板
>
> **写作原则**：每一节末尾尽量给"开发者备忘 (Dev Note)"，把"读完该做什么"明确化。
>
> **创建日期**：2026-04-27　|　**版本**：v0.1 (初稿)

---

## 第 0 章 速读 (One-Page Cheatsheet)

| 维度 | 内容 |
|---|---|
| **一句话定义** | 一种把 Agent 记忆组织成 Event → Episode → Pattern → Principle 四级抽象，并通过"预测-惊讶"信号自演化更新的记忆系统。 |
| **核心创新** | (C1) 多层级抽象栈；(C2) Predictive-Surprise 触发更新；(C3) Beta(α,β) 贝叶斯置信度；(C4) Utility×Stability 双轴遗忘；(C5) "对抗性反驳"诊断基准。 |
| **方法关键词** | Predictive coding, Bayesian online update, Hierarchical memory, Skill consolidation |
| **理论靠山** | Free-Energy Principle (Friston)、Predictive Coding (Rao & Ballard)、Complementary Learning Systems (McClelland)、在线贝叶斯 |
| **目标基准** | ALFWorld (主)、ScienceWorld、WebShop；自建 ALFWorld-Refute |
| **主要 Baseline** | No-Mem、Reflexion、Generative Agents (反思版)、MemGPT、A-MEM、Mem0、Agent Workflow Memory |
| **基础模型** | Qwen2.5-7B/14B-Instruct (主)、Qwen2.5-Coder-7B (技能/代码段)、GPT-4o (oracle) |
| **算力下限** | 单卡 24GB (4090/A5000) 起步；主表用 1×A100 80GB 更稳 |
| **周期** | 12 周 (4 周打基础 + 4 周做核心 + 4 周实验+写作) |
| **目标会议** | NeurIPS / ICLR / ACL Main (CCF-A) |

---

## 第 1 章 选题深化

### 1.1 问题陈述

LLM Agent 的"记忆"被广泛使用 (MemGPT、A-MEM、Mem0、Reflexion、Generative Agents、Agent Workflow Memory)，但这些系统在底层基本是 **"写入-检索"** 模型：一旦被写入，记忆条目就近似 **静态对象**。这造成三个具体问题：

**P1（无修订）** 记忆条目不会因后续证据被反驳/更新。例如，agent 第一次发现 "在 WebShop 上点击 'Buy' 按钮直接成单" 写入记忆；后续某次因为账号未登录，"Buy" 实际跳转到登录页——这条记忆仍按原样被检索，导致重复试错。

**P2（无抽象层级）** 多数系统只有"原始 trace"或"自由文本反思"两种粒度，缺少中间层 (recurrent pattern) 和顶层 (cross-task principle)。结果：要么检索过多原始细节噪声大，要么过度泛化丢失上下文。

**P3（无遗忘机制）** 记忆库随交互单调增长，检索成本和噪声同步上升；个别"过时但高频"的条目反而成为反向锚点。

P1–P3 在认知科学和神经科学里早有对应模型 (predictive coding、互补学习系统 CLS、schema theory)，但 **目前几乎没有 Agent 工作显式以这些理论为指导设计记忆机制**。

### 1.2 研究意义

- **学术意义**：把"记忆"从被动检索器升级为可演化的认知结构，对接神经科学/认知科学经典框架，为 LLM Agent 长程任务提供新的记忆范式。
- **工程意义**：在不重训模型的前提下，通过外部记忆系统实现持续自进化，对真实部署 (网页/操作系统/客服) 极为友好。
- **研究外溢**：本工作产出的 "ALFWorld-Refute" 反驳诊断基准本身可独立贡献给社区，弥补当前缺乏"记忆质量"评估的空白。

### 1.3 与现有工作的精细对比

> **撰写论文 Related Work 时直接复用本表。**

| 工作 | 记忆抽象层级 | 是否更新已有条目 | 是否有遗忘 | 是否处理矛盾证据 | 信号来源 |
|---|---|---|---|---|---|
| **Reflexion** (NeurIPS'23) | 单层 (verbal reflection) | ✗ (新轨迹追加) | ✗ | ✗ | 任务终态 reward |
| **Generative Agents** (UIST'23) | 双层 (memory stream + reflection) | △ (新 reflection 追加) | △ (importance 分数) | ✗ | 周期性触发 |
| **MemGPT** (2023) | 双层 (main ctx + archival) | △ (主体追加) | ✗ (paging 不算遗忘) | ✗ | 上下文溢出 |
| **A-MEM** (2025) | 单层 + 链接结构 | △ (链接更新非内容更新) | △ | ✗ | 任务相关性 |
| **Mem0** (2024) | 双层 (raw + extracted) | ✗ (写入即固定) | ✗ | ✗ | LLM 抽取 |
| **HippoRAG** (2024) | 双层 (passage + graph) | ✗ | ✗ | ✗ | 检索 PageRank |
| **Agent Workflow Memory** (2024) | 双层 (trace + workflow) | ✗ | ✗ | ✗ | 任务完成 |
| **本工作 RM** | **四层 (Event/Episode/Pattern/Principle)** | **✓ (Bayesian posterior 更新)** | **✓ (utility×stability)** | **✓ (Predictive-Surprise)** | **预测误差** |

**关键差异化**：本工作是首个把 **"记忆条目本身可被反驳/演化"** 作为一等公民、并以 predictive coding 提供机制基础的系统。其它工作即使有"反思"或"importance"，也只在写入时计算一次，之后凝固。

### 1.4 核心假设 (Hypotheses)

- **H1 (层级假设)**：分层抽象 (Event→Episode→Pattern→Principle) 比扁平存储在长程任务上有显著收益，因为高层条目浓缩了大量低层细节。
- **H2 (预测假设)**：用记忆做"自上而下预测"得到的 Surprise 信号，是更新记忆的 **充分信号**——即不需要外部 verifier 也能驱动更新。
- **H3 (反驳假设)**：当环境/分布发生变化时，本系统能比基线更快地更新或废弃过时条目。
- **H4 (容量假设)**：在性能持平时，本系统的记忆体积显著小于持续追加式系统 (Reflexion / Mem0)。

> **Dev Note**：H1–H4 各对应一个可定量验证的实验 (见 §4)。论文的故事线 = 4 个假设依次被证。

### 1.5 创新点拆解 (Contributions)

- **C1 四层记忆栈**：明确定义 Event / Episode / Pattern / Principle 的语义、Schema 和层间转化规则 (§2.2)。
- **C2 Predictive-Surprise 更新机制**：用上层记忆对下层观察做预测，预测误差作为更新触发器与权重 (§2.5)。
- **C3 Beta(α,β) 贝叶斯置信度**：每个 Pattern/Principle 维护 Beta 分布，支持/反驳证据各自更新 α/β，实现概率化的"信念修订"(§2.5.2)。
- **C4 Utility × Stability 双轴遗忘**：基于"被有效检索的频次 × 置信度稳定性 × 时间衰减"的可解释遗忘规则 (§2.6)。
- **C5 ALFWorld-Refute 诊断基准**：在 ALFWorld 上人为修改 affordance/规则，专门测试记忆的"反驳吸收"能力 (§4.5)。

---

## 第 2 章 方法详细设计

### 2.1 总体架构

```
                 ┌──────────────────────────────────────┐
                 │            Agent Main Loop           │
                 │  (ReAct: Thought → Action → Obs)     │
                 └──────────┬─────────────────┬─────────┘
                            │ write           │ retrieve
                            ▼                 ▼
        ┌────────────────────────────────────────────────┐
        │            Reflective Memory (RM)              │
        │                                                │
        │   ┌──────────┐ promote   ┌──────────┐         │
        │   │ Principle│ ◀──────── │ Pattern  │         │
        │   └────▲─────┘           └────▲─────┘         │
        │        │ refine               │ refine        │
        │        │ (surprise+Bayes)     │ (cluster+LLM) │
        │        │                      │               │
        │   ┌────┴───┐ summarize   ┌────┴───┐           │
        │   │Episode │ ◀────────── │ Event  │           │
        │   └────────┘             └────────┘           │
        │                                                │
        │   ── Storage: SQLite (struct) + Qdrant (vec) ──│
        │   ── Updater / Retriever / Forgetter modules ──│
        └────────────────────────────────────────────────┘
                            ▲                 │
                            │                 │ predict
                            │                 ▼
                 ┌──────────┴─────────────────┐
                 │  Predictive-Surprise Engine │
                 │  (LLM-as-judge or embed-Δ)  │
                 └────────────────────────────┘
```

数据流：
1. **写入流 (自下而上)**：Event → 周期性 LLM 切分 → Episode → 聚类+LLM 抽取 → Pattern → 多 Pattern 收敛 → Principle。
2. **检索流 (自上而下)**：查询时先检 Principle (cheap, few)，命中后展开相关 Pattern，必要时下钻 Episode。
3. **更新流 (横向反馈)**：Predictive-Surprise Engine 用现有 Pattern 预测新轨迹的关键步，比对实际结果，按 Beta 更新置信度，必要时触发 Pattern 重写。
4. **遗忘流 (周期触发)**：Forgetter 按 utility×stability 评分，purge / consolidate 低分条目。

### 2.2 四层记忆栈 — Schema 与语义

> 以下 Schema 可直接转成 Pydantic 模型 (§附录 C)。

#### 2.2.1 Event 层

```yaml
Event:
  event_id: str            # uuid
  trajectory_id: str
  step_idx: int
  state: str               # 自然语言描述当前观察
  action: str              # agent 输出的 action 文本
  observation: str         # 环境返回的 observation
  reward: float | None
  ts: datetime
  embedding: list[float]   # 用 BGE-M3 / Qwen3-Embedding 等
```

- **作用**：原始 trace；其它层都从 Event 派生。
- **生命周期**：永久 (但可压缩归档至冷存储)；只有当所属 trajectory 已被完全 Episode 化后允许压缩。

#### 2.2.2 Episode 层

```yaml
Episode:
  episode_id: str
  trajectory_id: str
  start_step: int
  end_step: int
  sub_goal: str            # LLM 标注的子目标
  summary: str             # 1-3 句话摘要
  outcome: enum {success, partial, failure}
  key_steps: list[int]     # 子目标内的关键 event_id
  embedding: list[float]
```

- **作用**：把长 trajectory 切成"有意义的子任务"块；Pattern 抽取的最小单位。
- **生成时机**：trajectory 结束时 (offline) 或子任务完成检测命中时 (online)。

#### 2.2.3 Pattern 层 (★核心)

```yaml
Pattern:
  pattern_id: str
  condition: str           # "当 ... 时"，自然语言谓词
  action_template: str     # 可参数化的行为草案
  expected_effect: str     # 期望观察/状态变化
  scope: list[str]         # 适用任务族标签
  support_episodes: list[str]
  refute_episodes: list[str]
  alpha: float = 1.0       # Beta 后验 α
  beta: float = 1.0        # Beta 后验 β
  evidence_count: int = 0
  last_updated: datetime
  version: int             # 每次重写 +1
  embedding: list[float]
```

- **置信度**：`p(success | pattern) ~ Beta(α, β)`，后验均值 = α/(α+β)，方差用作不确定性。
- **生成时机**：当某 Episode 簇 ≥ K 条 (默认 K=3) 时触发 LLM 抽取。

#### 2.2.4 Principle 层

```yaml
Principle:
  principle_id: str
  statement: str           # 一句格言式陈述
  scope: str               # cross-task / domain-wide
  supporting_patterns: list[str]
  contradiction_log: list[dict]  # {pattern_id, ts, brief}
  alpha: float = 1.0
  beta: float = 1.0
  embedding: list[float]
```

- **作用**：跨任务的"认知偏置"；用于 prompt 顶部的"系统提示"。
- **生成时机**：Patterns 出现 (a) 跨任务族收敛 或 (b) 反复矛盾时，由元反思流程产生。

> **Dev Note**：Pattern 是最有"研究价值"的层；Episode/Event 偏工程；Principle 是亮点 (在论文里展示几条诱人的 Principle，比如 "Always check side-effect reversibility before destructive actions" 是非常好的 figure 素材)。

### 2.3 写入流程：自下而上抽象

```python
# 伪代码
def on_step(event: Event):
    store.write_event(event)

def on_trajectory_end(traj_id):
    events = store.get_events(traj_id)
    episodes = LLM_segment(events)             # §2.8 Prompt P1
    for ep in episodes:
        store.write_episode(ep)
    
    # 检查是否触发 Pattern 抽取
    candidate_clusters = cluster_recent_episodes(k=K_PATTERN)
    for cluster in candidate_clusters:
        if cluster.size >= MIN_SUPPORT:
            new_pattern = LLM_induce_pattern(cluster)   # §2.8 Prompt P2
            store.write_pattern(new_pattern)

def periodic_principle_reflection(every=N_TRAJ):
    patterns = store.get_recent_patterns(N=200)
    converging, conflicting = analyze(patterns)
    new_principles = LLM_induce_principles(converging, conflicting)  # P3
    for p in new_principles:
        store.write_principle(p)
```

- **聚类策略**：在 Episode 嵌入上做 HDBSCAN (无需指定 k) 或 mini-batch K-means；默认 HDBSCAN，min_cluster_size=3。
- **去重策略**：写入新 Pattern 前查询近邻 Pattern (cosine > 0.9) → 触发"是否合并"判断而非新增。

### 2.4 检索流程：自上而下查询

```python
def retrieve(query: str, k_principle=3, k_pattern=5, k_episode=3):
    q_emb = embed(query)
    
    # Layer 1: Principles (always include high-confidence)
    principles = store.query_principles(q_emb, top_k=k_principle, 
                                        min_confidence=0.6)
    
    # Layer 2: Patterns (filtered by current task scope)
    scope_tags = infer_scope(query)
    patterns = store.query_patterns(q_emb, top_k=k_pattern, 
                                    scope=scope_tags,
                                    min_confidence=0.5)
    
    # Layer 3: Episodes (case-based, only if low-confidence Patterns)
    if avg_confidence(patterns) < CONF_THRESH:
        episodes = store.query_episodes(q_emb, top_k=k_episode)
    else:
        episodes = []
    
    return MemoryContext(principles, patterns, episodes)
```

- **拼接到 Prompt 的顺序**：System (Principle) → 上下文片 (Pattern) → 案例 (Episode)。
- **token 预算**：默认每层最多 1500 tokens，超出按置信度截断。

### 2.5 更新机制 — Predictive-Surprise + Bayesian (★核心创新)

#### 2.5.1 Predictive-Surprise 计算

对新到来的 Episode `e_new`：

```python
def compute_surprise(e_new, retrieved_patterns):
    surprises = []
    for p in retrieved_patterns:
        # 用 LLM 让 Pattern p 对 e_new 的关键步做"事前预测"
        predicted = LLM_predict(p, e_new.context_prefix)  # P4
        actual = e_new.observed_outcome
        # 用 LLM-as-judge 给 0-1 分歧度
        s = LLM_judge_divergence(predicted, actual)  # P5
        surprises.append((p, s))
    return surprises
```

**实现选项 (按优先级)**：
1. **LLM-as-judge** (默认)：用同一 LLM 出 1–5 离散分歧度 → 归一到 [0,1]。优点：语义敏感；缺点：贵+噪声。
2. **嵌入距离**：cos_dist(embed(predicted), embed(actual))。优点：便宜；缺点：表面化。
3. **条件困惑度**：log p(actual | context+predicted) − log p(actual | context)，其中 p 用基础 LLM。优点：信息论清晰；缺点：需 logprobs。

> **Dev Note**：先用方案 1 跑通主表，方案 3 作为消融变体 (展示 surprise 度量的鲁棒性)。

#### 2.5.2 Beta 贝叶斯更新

```python
def bayesian_update(pattern, surprise_score):
    # 阈值化
    if surprise_score < TAU_LOW:    # 0.2: 强支持
        pattern.alpha += 1
        pattern.support_episodes.append(e_new.episode_id)
    elif surprise_score > TAU_HIGH: # 0.7: 强反驳
        pattern.beta += 1
        pattern.refute_episodes.append(e_new.episode_id)
    else:
        # 模糊证据：按比例软更新
        pattern.alpha += (1 - surprise_score)
        pattern.beta += surprise_score
    
    pattern.evidence_count += 1
    pattern.last_updated = now()
    
    # 触发重写：连续 N 条强反驳
    if recent_refutes(pattern, window=5) >= REWRITE_THRESH:
        new_p = LLM_revise_pattern(pattern, refute_ep=...)  # P6
        store.upsert_pattern(new_p)  # version += 1; 重置 α, β
```

**Beta 选择的理由**：
- 共轭先验，更新闭式简单。
- 后验均值就是直观的"成功率"。
- 方差给出不确定性，便于 retrieval 时做置信度过滤。

> **Dev Note**：阈值 TAU_LOW/HIGH/REWRITE_THRESH 全部走配置文件，便于消融。

#### 2.5.3 Pattern 重写 (Revision)

当 Pattern 被持续反驳，说明它在新分布下失效。两种修订方式：
- **Refine**：保留 condition，仅修改 action_template / expected_effect (小改)。
- **Split**：用 LLM 判断，如果反驳证据来自不同子情境，则把原 Pattern 拆为多个子 Pattern (大改，version 重置)。

### 2.6 遗忘机制 — Utility × Stability 双轴

每个记忆条目 m 定义：

```
utility(m)   = exp(−λ_u · (now − last_used)) · Σ (success_contribution_i)
stability(m) = α / (α + β)                    # Pattern/Principle
             = recent_access_freq             # Event/Episode

forget_score(m) = w1 · (1 − utility(m)) + w2 · (1 − stability(m)) + w3 · age(m)
```

每周期触发 (例如每 N=50 trajectories)：
- 排序 forget_score；
- 取 top-K 比例的条目执行：
  - 若条目可与近邻 (cosine > θ_merge) **合并** → consolidate；
  - 否则 → archive (移到冷存储，不参与检索)。

> **Dev Note**：合并优先于删除。论文里强调"我们不丢信息，只是降权 + 浓缩"。

### 2.7 Agent 主循环接口

```python
class ReflectiveAgent:
    def __init__(self, llm, memory: ReflectiveMemory, env):
        self.llm = llm; self.memory = memory; self.env = env
    
    def run_episode(self, task):
        traj = []
        obs = self.env.reset(task)
        while not done:
            ctx = self.memory.retrieve(query=task + " | " + obs)
            prompt = build_react_prompt(task, obs, ctx, traj)
            thought, action = self.llm.generate(prompt)
            new_obs, reward, done = self.env.step(action)
            event = Event(state=obs, action=action, observation=new_obs, ...)
            self.memory.write_event(event)
            traj.append(event)
            obs = new_obs
        
        # 写入后处理
        self.memory.on_trajectory_end(traj_id=traj[0].trajectory_id)
        return traj, reward
```

### 2.8 关键 Prompt 模板 (节选；完整版见附录 B)

#### P1 — Episode 切分

```
你是一个轨迹分析师。下面是一段 Agent 完成任务的 step 序列。
请把它切分为若干"语义连贯的子目标块"。每块输出 JSON:
  {sub_goal: str, start_step: int, end_step: int, summary: str, outcome: success/partial/failure}
要求：相邻 step 的 sub_goal 不同则切分；每块至少 2 step；忽略明显的无效探索。

Trace:
[step 0] state=... action=... obs=...
[step 1] ...
...
```

#### P2 — Pattern 抽取

```
你是一个行为模式归纳专家。下面是 K 段相似的 Episode，它们解决相似子目标。
请抽取一个"行为模式 (Pattern)"。输出 JSON:
  {condition: 一段自然语言谓词, action_template: 可参数化模板,
   expected_effect: 期望发生的状态变化, scope: 任务族标签}
要求：condition 要可被未来情境复用；action_template 要避免过度具体的字面量。

Episodes:
... (K 段)
```

#### P4 — Pattern 做预测

```
给定一个行为模式：
  Condition: ...
  Action: ...
  Expected effect: ...
以及当前情境前缀：...
请预测：如果按该 Pattern 行动，应该观察到什么？输出一句话。
```

#### P5 — 分歧度评判

```
预测：...
实际：...
任务：判断两者的语义分歧程度。输出 1-5 的整数：
1=几乎完全一致；2=细节不同；3=部分一致；4=明显冲突；5=完全相反。
仅输出数字。
```

#### P6 — Pattern 修订

```
旧 Pattern: ...
反驳证据 (3-5 段 Episode): ...
分析：旧 Pattern 在哪些情境下失效？
请输出：(a) 修订建议 refine/split (b) 新的 Pattern JSON (一个或多个)。
```

> **Dev Note**：Prompt 是论文复现性的命脉。把所有 prompt 版本化 (`prompts/v1/*.txt`)，每次大改 bump 版本号并在表里标注。

---

## 第 3 章 理论支撑与论证

### 3.1 形式化定义

设 Agent 在环境 ε 中产生轨迹 τ = (s_0, a_0, ..., s_T)，记忆 M = {Event, Episode, Pattern, Principle} 四集合。

**Pattern 作为预测模型**：每个 Pattern p_i 是一个条件分布 p_i(o | s, a)，对给定 (s,a) 给出期望观察 o 的分布。Beta(α_i, β_i) 是该 Pattern 在过去证据下成立的元分布。

**Surprise 定义**：对新观察 (s,a,o)，
```
surprise_i = − log p_i(o | s, a)
```
近似实现为 `LLM_judge(predicted, actual) ∈ [0, 1]`。

**贝叶斯更新**：观察到证据 e 后，
```
α_i' = α_i + (1 − surprise_i)
β_i' = β_i +     surprise_i
```
(soft 更新；阈值版本对应 0/1 硬更新。)

### 3.2 Predictive-Surprise 的信息论解释

定义 RM 的"自由能"为 (借鉴 Friston):
```
F(M) = E_q[− log p(o | M)] + KL(q(M) || p(M))
     = (negative log-likelihood) + (complexity)
```
- 第一项：M 对观察的预测能力 (越大越差)。
- 第二项：M 的复杂度 (越大越冗余)。

**核心声明**：本系统的更新流程 = 近似最小化 F(M)：
- Bayesian update 减小第一项 (后验对当前数据拟合更好)；
- Forgetting & consolidation 减小第二项 (压缩复杂度)。

> 这是写论文 §3 时最有力的"理论一段话"。Friston 的自由能原理是公认的 cognitive theory，能给方法一个高大上的元解释，且不需要严格数学证明。

### 3.3 收敛性 sketch (适合作为 §A.1 附录)

在以下假设下：
- (A1) 环境是分段平稳的 (piecewise stationary)；
- (A2) Surprise 估计无偏 (E[surprise_i] = 真实预测误差)；
- (A3) Bayesian 更新 + α/β 上界限制 (避免无限累积)，

可证：在每个平稳段内，Pattern 的后验均值收敛到 真实的条件成功率。当分布发生变化时，由于 Surprise 上升触发 revision，系统的"最近后验"在 O(1/√N) 内重新收敛 (N = 新段内观察数)。

> **Dev Note**：不必给完整证明；3 步骤 + 引用 (在线学习经典结果) 即可。完整证明放附录。

### 3.4 复杂度

| 操作 | 时间 | 空间 |
|---|---|---|
| 写 Event | O(1) + 1 embed call | O(\|E\|) |
| Episode 切分 | 1 LLM call / trajectory | O(\|Eps\|) |
| Pattern 抽取 | 1 LLM call / cluster | O(\|P\|) |
| 检索 (一次) | 3 × ANN query | — |
| 更新 (per step) | 1 LLM judge call × top-k pattern | — |
| 反思 (周期) | 1 LLM call / N trajectories | — |

主要成本在 LLM call。一次完整 trajectory 的 RM overhead 大致 = 6–10 倍 base agent。可接受 (Reflexion 也类似)。

---

## 第 4 章 实验设计

### 4.1 Benchmark 选择

| 基准 | 选用理由 | 入门成本 | 备注 |
|---|---|---|---|
| **ALFWorld** | 文本化家居任务，记忆消融文献"标准件"，单 step 廉价；6 类任务、134 测试 | 低 | 主表必须；Reflexion 等都用它 |
| **ScienceWorld** | 30 类科学实验任务，长程序更长 (50–100 步)，有数值化进度 | 低 | 主表第二；测长程能力 |
| **WebShop** | 半真实商品浏览/购买，含搜索/筛选，记忆效益明显 | 中 | 主表第三；Mem0/A-MEM 都用 |
| **WebArena (subset)** | 真实 web 操作 (Reddit/GitLab/CMS)，难度高 | 高 | stretch goal；放在 §5 "extended" |
| **OSWorld** | 桌面应用 (LibreOffice 等)，视觉+API 双模态 | 极高 | 不主推，时间够再做 |

**主表用 3 个 (ALFWorld + ScienceWorld + WebShop)**；这是足以构成 CCF-A 主表的最小集合。

### 4.2 Baseline 列表与复现策略

| Baseline | 类别 | 公开实现 | 复现策略 | 难度 |
|---|---|---|---|---|
| **No-Mem** | 下界 | — | 直接写一个 vanilla ReAct | 极低 |
| **Reflexion** (Shinn'23) | 单层反思 | github.com/noahshinn/reflexion | fork → 换模型 → 在 ALFWorld 复刻表内数 (容差 ±2pp) | 低 |
| **Generative Agents** (Park'23) | 双层反思 | joonspk-research/generative_agents | 抽取 reflection 模块 + memory stream，封装到 agent loop | 中 |
| **MemGPT / Letta** | 双层 + paging | github.com/cpacker/MemGPT | 用其 SDK 封装为 ALFWorld agent | 中 |
| **A-MEM** | 单层 + 链接 | github.com/agiresearch/A-mem | fork → 适配三个 env | 中 |
| **Mem0** | 双层 (raw + extracted) | github.com/mem0ai/mem0 | 直接用其 Python SDK | 低 |
| **Agent Workflow Memory** | 双层 (trace + workflow) | github.com/zorazrw/agent-workflow-memory | 适配；其原 paper 用 WebArena | 中 |

**复现 Sanity Gate**：先用 GPT-3.5/4 复现 Reflexion 论文 ALFWorld 的成功率到 ±2pp。**这是 Week 2 必须达成的硬指标**——通不过就先排查环境/prompt，不要往后做。

### 4.3 主表设计

```
                                ALFWorld     ScienceWorld     WebShop
                                ────────     ────────────     ───────
                                SR    Steps    SR    Steps    SR    Steps
No-Mem (Qwen2.5-7B)             ..    ..       ..    ..       ..    ..
+ Reflexion                     ..    ..       ..    ..       ..    ..
+ Generative Agents (refl)      ..    ..       ..    ..       ..    ..
+ MemGPT                        ..    ..       ..    ..       ..    ..
+ A-MEM                         ..    ..       ..    ..       ..    ..
+ Mem0                          ..    ..       ..    ..       ..    ..
+ AWM                           ..    ..       ..    ..       ..    ..
+ RM (Ours)                     ..    ..       ..    ..       ..    ..
                            ─────────────────────────────────────
+ RM (GPT-4o oracle)            ..    ..       ..    ..       ..    ..
```

- **运行 N = 3 seeds**，报告均值 ± 标准差；统计显著性用 paired bootstrap。
- **每个 task suite 的 3 trial 设置**：参照 Reflexion，agent 有 3 次尝试同一任务，记忆累积。

### 4.4 消融实验 (★必做)

| ID | 变体 | 验证假设 |
|---|---|---|
| A1 | RM w/o Episode (跳过 §2.2.2) | H1 — 中间层必要性 |
| A2 | RM w/o Pattern (Episode → Principle) | H1 — Pattern 层必要性 |
| A3 | RM w/o Principle | H1 — 顶层必要性 |
| A4 | RM w/o Predictive-Surprise (硬写不更新) | H2 — 更新机制必要性 |
| A5 | RM w/o Bayesian (用 hard count 代替 α/β) | C3 内部 — Bayesian 必要性 |
| A6 | RM w/o Forgetting | H4 — 容量假设 |
| A7 | Surprise 度量改用 embed-Δ | C2 内部 — surprise 度量鲁棒性 |
| A8 | Surprise 度量改用 logprob | 同上 |
| A9 | 替换 base LLM (Qwen2.5-7B → 14B) | 模型规模敏感性 |
| A10 | Embedder 替换 (BGE-M3 vs Qwen3-Embedding) | 检索敏感性 |

### 4.5 诊断实验：ALFWorld-Refute (★亮点)

**动机**：常规基准无法测出"记忆是否能修订"，因为环境平稳。我们构造一个"反驳测试"。

**构造方法**：
- 取 ALFWorld 的标准 train，让 agent 充分学习，形成 Pattern。
- 构造测试集：手工/脚本修改环境规则，例如：
  - 原本 "drawer 默认关闭，需 open 后再 take" → 现在 "drawer 默认开启，open 会反而关上"
  - 原本 "microwave 加热食物 → hot" → 现在 "microwave 损坏，加热返回 cold"
- 让 agent 在修改后的环境跑 N 个 trajectory；
- 观察其 Pattern 库 P_t 的演化。

**评估指标 (本工作首倡)**：
- **Refutation Lag (RL)**：从首次反驳证据出现到 Pattern 被修订/废弃的 trajectories 数。
- **Belief Revision Accuracy (BRA)**：修订后 Pattern 是否反映新规则 (LLM-judge + 人工抽检)。
- **Stale-Memory Penalty (SMP)**：测试期内被过时 Pattern 误导的步数比例。

**预期表**：
```
                Refutation Lag↓   BRA↑    SMP↓
No-Mem               N/A          N/A     N/A
Reflexion            ∞ (不更新)    0%      高
Mem0                 ∞            0%      高
A-MEM                ∞ (内容不更新) 0%      高
RM (Ours)            ~5–10        >70%    显著低
```

**这张表是论文的"杀手锏 figure"**——它证明前人方法在分布漂移下都崩，而我们不崩。

### 4.6 评估指标汇总

- **任务成功率 (SR)**：standard。
- **平均完成步数 (Steps)**：长程效率。
- **Token 消耗 (Tokens/task)**：成本。
- **记忆体积 (\|M\|)**：随 trajectory 数的曲线 (体现 H4)。
- **跨任务族迁移 (Transfer SR)**：在 task family A 上学习，测 family B。
- **Refutation Lag / BRA / SMP**：仅 §4.5。

### 4.7 失败模式与对照表

预先想清楚 reviewer 会怎么挑刺：

| 风险 | 对策 |
|---|---|
| "贵" — 计算成本远高于 baseline | 做"per-task token cost"对照；并报告 amortized cost (随 trajectory 数增长，cost/task 下降) |
| "Surprise 由 LLM 自打分，会不会 reward hacking" | A7/A8 消融；并在主表加 GPT-4o judge 第三方校验 |
| "Pattern 抽取依赖 LLM，质量靠 prompt" | 提供 prompt 版本化、人工抽检准确率、ensemble 投票 |
| "新基准 ALFWorld-Refute 是不是 cherry-picked" | 公开构造脚本；并在 ScienceWorld 上构造类似 -Refute 子集做交叉验证 |
| "为什么不 RL 微调" | 强调 zero-training 的工程意义 + 与微调正交，可叠加 |

---

## 第 5 章 编码路线图

### 5.1 技术栈 (推荐版本)

| 类别 | 选型 | 版本 | 备注 |
|---|---|---|---|
| Python | 3.10+ | — | type hint 完整支持 |
| 包管理 | uv | latest | 比 pip/poetry 快 10×+ |
| LLM 推理 (本地) | vLLM | ≥0.6.x | 单卡 7B/14B；OpenAI-compat API |
| LLM 推理 (备份) | OpenAI/Anthropic SDK | latest | 接 GPT-4o、Claude oracle |
| 向量库 | Qdrant (docker) | ≥1.10 | 比 Chroma 稳；支持 payload filter |
| 关系存储 | SQLite + SQLAlchemy | — | 轻量；schema migration 用 Alembic |
| Embedder | BGE-M3 / Qwen3-Embedding-0.6B | latest | 多语言、便宜 |
| Schema | Pydantic v2 | ≥2.7 | 强 typing |
| 配置 | Hydra | ≥1.3 | 嵌套 config，便于消融 |
| 跟踪 | Weights & Biases | latest | 实验对比 |
| 测试 | pytest | latest | — |
| Agent 框架 | (vanilla / DSPy / AgentScope) | — | 推荐 vanilla 起步，避免框架黑盒 |
| Env | alfworld / scienceworld / webshop | 各自最新 | 见下 |

> **Dev Note**：避免提早引入 LangChain/LlamaIndex 这种重抽象框架，会让你看不清自己在做什么。直接写 vanilla 的 ReAct loop，顶多用 Pydantic 管 schema。

### 5.2 仓库结构

```
reflective-memory/
├── pyproject.toml
├── README.md
├── docs/
│   ├── RM_design_and_roadmap.md   ← 本文档
│   ├── prompts_v1.md
│   └── adr/                        ← Architecture Decision Records
├── configs/
│   ├── base.yaml
│   ├── env/{alfworld,scienceworld,webshop}.yaml
│   ├── agent/{noMem,reflexion,rm}.yaml
│   ├── llm/{qwen7b,qwen14b,gpt4o}.yaml
│   └── exp/{main,ablation,refute}.yaml
├── src/rm/
│   ├── __init__.py
│   ├── memory/
│   │   ├── schemas.py        # Pydantic models for Event/Episode/Pattern/Principle
│   │   ├── store.py          # SQLite + Qdrant 抽象
│   │   ├── writer.py         # 写入 + Episode/Pattern/Principle 抽取
│   │   ├── retriever.py      # 三层 retrieval
│   │   ├── updater.py        # Predictive-Surprise + Bayesian
│   │   ├── forgetter.py      # utility × stability
│   │   └── llm_ops.py        # 所有 LLM 调用 (segment/induce/predict/judge/revise)
│   ├── agent/
│   │   ├── base.py
│   │   ├── react.py          # 通用 ReAct
│   │   └── reflective.py     # 用 RM 的 agent
│   ├── envs/
│   │   ├── base.py           # 统一 step()/reset() 接口
│   │   ├── alfworld_env.py
│   │   ├── scienceworld_env.py
│   │   └── webshop_env.py
│   ├── baselines/
│   │   ├── reflexion.py
│   │   ├── memgpt.py
│   │   ├── amem.py
│   │   ├── mem0_wrapper.py
│   │   ├── awm.py
│   │   └── gen_agents.py
│   ├── llm/
│   │   ├── client.py         # 统一 LLM client (vllm / openai / anthropic)
│   │   ├── prompts/          # 版本化 prompt
│   │   │   ├── v1/
│   │   │   │   ├── P1_segment.txt
│   │   │   │   ├── P2_pattern.txt
│   │   │   │   ├── ...
│   │   └── embed.py
│   ├── eval/
│   │   ├── runner.py         # 实验入口
│   │   ├── metrics.py        # SR / Steps / Tokens / |M| / Transfer
│   │   ├── stats.py          # bootstrap 显著性
│   │   └── refute.py         # ALFWorld-Refute 评估
│   └── utils/
│       ├── logging.py
│       └── seeding.py
├── scripts/
│   ├── 00_smoke_alfworld.py     # 跑通 ALFWorld + Qwen
│   ├── 01_repro_reflexion.py    # 复现 Reflexion 数字
│   ├── 02_run_baseline.py
│   ├── 03_run_rm.py
│   ├── 04_run_ablation.py
│   ├── 05_build_refute_env.py   # 构造 ALFWorld-Refute
│   └── 06_run_refute.py
├── tests/
│   ├── test_schemas.py
│   ├── test_store.py
│   ├── test_updater.py
│   └── test_retriever.py
├── data/
│   ├── alfworld_refute/         # 修改后的环境定义
│   └── traces/                  # 缓存的 trajectory 用于 reproduce
└── notebooks/
    ├── 01_explore_alfworld.ipynb
    └── 02_inspect_memory.ipynb
```

### 5.3 模块依赖图

```
        agent.reflective ──── memory.* ──── llm.client
              │                  │              │
              │                  └─ embed ──────┤
              │                                 │
        envs.* ─────────────────────────────────┤
              │                                 │
        eval.runner ───── metrics ──────────────┘
```

**关键不变量**：`memory.*` 内部只能依赖 `llm.client` 和 `schemas`；不能依赖 `envs`/`agent`。这是把"记忆"做成"可独立测试"的前提。

### 5.4 12 周里程碑

| 周 | 主题 | 交付物 |
|---|---|---|
| **W1** | 环境 + Sanity | vLLM 跑通、ALFWorld 跑通、随机/ReAct agent 在 5 个 task 上能 step 完 |
| **W2** | Reflexion 复现 | scripts/01 输出与 Reflexion 论文 ±2pp 一致；W&B 看板成型 |
| **W3** | RM 骨架 v0 | schemas+store+retriever 完成；只用 Event 层做"案例式"记忆，agent 接得上 |
| **W4** | Episode + Pattern | writer 完成；HDBSCAN 聚类 + LLM 抽取；测试集上能看到 reasonable Pattern |
| **W5** | Predictive-Surprise | updater 完成；P4/P5 跑通；记录 surprise 分布直方图 |
| **W6** | Bayesian + Revision | Beta 更新 + Pattern 重写跑通；写入 8–10 个单测 |
| **W7** | Principle + Forgetting | 周期反思 + 双轴遗忘；端到端 RM v1.0 在 ALFWorld 上不输 Reflexion |
| **W8** | 多基准 + Baseline 对齐 | ScienceWorld、WebShop 跑通；MemGPT/A-MEM/Mem0/AWM/GenAgents 全部能跑 |
| **W9** | 主表 + 消融 1 | 主表 SR/Steps/Tokens 出齐；A1–A4 消融完 |
| **W10** | 消融 2 + ALFWorld-Refute | A5–A10 完成；构造 Refute env 并跑出 RL/BRA/SMP |
| **W11** | 写作 + 反复跑实验 | 论文 8 页初稿；图表 v1；与导师 review |
| **W12** | 抛光 + 投稿 | rebuttal 实验 buffer；最终版；按目标会议改格式 |

> **缓冲建议**：W11 之后留至少 1 周给 "实验补做"。论文从来不是一次性写完的。

### 5.5 风险登记 (Risk Register)

| ID | 风险 | 概率 | 影响 | 缓解 |
|---|---|---|---|---|
| R1 | Reflexion 复现失败 (W2) | 中 | 高 | 早 (W1) 联系作者/查 issue；准备用 GPT-4o 替代 Qwen 跑一组对照 |
| R2 | Pattern 抽取质量差 | 中 | 高 | 多版本 prompt + 人工抽检 50 条；ensemble 投票 |
| R3 | Surprise 度量不可靠 | 中 | 中 | A7/A8 消融兜底 |
| R4 | 算力不够 | 中 | 高 | 优先在 ALFWorld 上做完所有实验 (廉价)；ScienceWorld/WebShop 上做缩小 N |
| R5 | benchmark 分歧 (各 paper 报数不一致) | 高 | 中 | 全部以"自己复现的 Reflexion 数字"为锚点；论文里明说 |
| R6 | 与并发工作撞车 | 中 | 高 | 每周一查 arXiv (cs.AI/cs.LG/cs.CL，关键词：agent memory, self-evolving) |
| R7 | 写作时间不够 | 高 | 高 | W7 起就开始写 §3，不等实验跑完 |

### 5.6 预算估算

**算力**：
- 单卡 4090 (24GB): 跑 Qwen2.5-7B 主体实验约 600–900 GPU·小时
- 1×A100 80GB: 14B 消融约 200 GPU·小时
- 总：本地 ≈ 1k GPU 小时；云 ≈ ¥3k–8k (按现价)

**API**：
- GPT-4o 用作 oracle + ALFWorld-Refute LLM-judge：~ $200–500
- 备用 Claude/DeepSeek：~ $100

**总预算 (单人 12 周)**：¥5k–12k，可控。

---

## 第 6 章 Day-1 启动手册

### 6.1 环境搭建 (Day 1)

```bash
# 1. 创建 Python env (推荐 uv)
curl -LsSf https://astral.sh/uv/install.sh | sh   # Linux/macOS
# 或在 Windows PowerShell 用 winget install astral-sh.uv

uv venv --python 3.10
source .venv/bin/activate                # Windows 是 .venv\Scripts\activate

# 2. 安装基础依赖
uv pip install vllm pydantic hydra-core wandb sqlalchemy alembic \
               qdrant-client sentence-transformers \
               openai anthropic \
               pytest ruff mypy

# 3. 拉 Qdrant
docker run -d -p 6333:6333 qdrant/qdrant

# 4. 拉模型 (下载 Qwen2.5-7B-Instruct)
huggingface-cli download Qwen/Qwen2.5-7B-Instruct --local-dir ./models/qwen7b

# 5. 起 vLLM 服务
python -m vllm.entrypoints.openai.api_server \
    --model ./models/qwen7b \
    --port 8000 \
    --max-model-len 8192

# 6. 安装 ALFWorld
uv pip install alfworld
alfworld-download
```

### 6.2 Reflexion 复现 Checklist (Week 2 必过)

- [ ] Clone github.com/noahshinn/reflexion，跑通其 ALFWorld demo
- [ ] 替换其 OpenAI 调用为本地 vLLM (相同 OpenAI-compat API，改 base_url 即可)
- [ ] 用 GPT-3.5 跑论文设置 (134 unseen)，目标 SR ≈ 78–85% (论文报)
- [ ] 用 Qwen2.5-7B 跑同设置，记录 SR (作为本工作 baseline，不需要打到论文数；只要稳定即可)
- [ ] 把以上结果记到 W&B 一张 "baseline reproducibility" board

### 6.3 Week 1 具体任务 (按天拆分)

| 天 | 任务 |
|---|---|
| Mon | 环境 (§6.1)；hello-world Qwen2.5-7B 推理 (curl http://localhost:8000) |
| Tue | 跑通 ALFWorld；写 `scripts/00_smoke_alfworld.py`：random agent on 5 tasks |
| Wed | 写 `src/rm/envs/alfworld_env.py`：统一 step/reset/render 接口；写 1 个 pytest |
| Thu | 写 `src/rm/agent/react.py` + `src/rm/llm/client.py`；vanilla ReAct on 5 ALFWorld tasks |
| Fri | 跑 50 个 ALFWorld 任务，记录 baseline SR；初始化 W&B 项目 |
| Sat/Sun | 文献深读 (Reflexion / Generative Agents / A-MEM 三篇精读) + 写读书笔记 |

### 6.4 第一段值得执行的代码 (拷贝即用)

`src/rm/llm/client.py` (最小骨架):

```python
from openai import OpenAI
from typing import Optional

class LLMClient:
    def __init__(self, base_url="http://localhost:8000/v1",
                 api_key="EMPTY", model="qwen7b"):
        self.client = OpenAI(base_url=base_url, api_key=api_key)
        self.model = model

    def chat(self, messages, temperature=0.0, max_tokens=512,
             response_format: Optional[dict] = None):
        kwargs = dict(model=self.model, messages=messages,
                      temperature=temperature, max_tokens=max_tokens)
        if response_format:
            kwargs["response_format"] = response_format
        return self.client.chat.completions.create(**kwargs)
```

`scripts/00_smoke_alfworld.py`:

```python
import alfworld.agents.environment as environment
import yaml, random

with open("configs/alfworld.yaml") as f:
    config = yaml.safe_load(f)

env = environment.AlfredTWEnv(config, train_eval="eval_out_of_distribution")
env = env.init_env(batch_size=1)

obs, info = env.reset()
print("Initial obs:", obs[0][:500])
for _ in range(20):
    action = random.choice(info["admissible_commands"][0])
    obs, score, done, info = env.step([action])
    print(f"Action: {action} | Score: {score[0]} | Done: {done[0]}")
    if done[0]: break
```

跑通这两段，你就完成了"从零到第一帧"。

---

## 附录 A：关键参考文献清单 (建议精读 →)

> 标 **[★]** 为必读。

### Agent 自进化 / 记忆
- [★] Shinn et al., "Reflexion: Language Agents with Verbal Reinforcement Learning", NeurIPS 2023
- [★] Park et al., "Generative Agents: Interactive Simulacra of Human Behavior", UIST 2023
- Packer et al., "MemGPT: Towards LLMs as Operating Systems", 2023
- [★] Xu et al., "A-MEM: Agentic Memory for LLM Agents", 2025
- Chhikara et al., "Mem0: Building Production-Ready AI Agents", 2024
- Gutiérrez et al., "HippoRAG: Neurobiologically Inspired Long-Term Memory", 2024
- Wang et al., "Agent Workflow Memory", 2024
- Madaan et al., "Self-Refine: Iterative Refinement with Self-Feedback", NeurIPS 2023
- Wang et al., "Voyager: An Open-Ended Embodied Agent with LLMs", 2023

### 评估 / 基准
- Yao et al., "ALFWorld", ICLR 2021
- Wang et al., "ScienceWorld", EMNLP 2022
- Yao et al., "WebShop", NeurIPS 2022
- Zhou et al., "WebArena", ICLR 2024
- Xie et al., "OSWorld", NeurIPS 2024

### 理论靠山
- [★] Friston, "The free-energy principle: a unified brain theory?", Nat Rev Neurosci 2010
- Rao & Ballard, "Predictive coding in the visual cortex", Nat Neurosci 1999
- McClelland, McNaughton, O'Reilly, "Why there are complementary learning systems...", Psych Rev 1995
- Tulving, "Episodic and Semantic Memory", 1972

### 评估方法学 / 因果
- Koh & Liang, "Understanding Black-box Predictions via Influence Functions", ICML 2017
- Pearl, "Causality" (book), 2009

### 同期值得跟踪 (建议每周一查 arXiv 关键词)
- "self-evolving agents", "agent memory", "agent self-improvement",
  "lifelong agents", "predictive memory", "online belief revision in LLM"

---

## 附录 B：完整 Prompt 模板 (v1)

(占位：待 W3 开发时填充实际版本，并按 `prompts/v1/Pi_*.txt` 落盘)

---

## 附录 C：Pydantic Schema (可直接复制到 src/rm/memory/schemas.py)

```python
from pydantic import BaseModel, Field
from datetime import datetime
from typing import List, Optional, Literal
from uuid import uuid4

def _uuid() -> str:
    return str(uuid4())

class Event(BaseModel):
    event_id: str = Field(default_factory=_uuid)
    trajectory_id: str
    step_idx: int
    state: str
    action: str
    observation: str
    reward: Optional[float] = None
    ts: datetime = Field(default_factory=datetime.utcnow)
    embedding: Optional[List[float]] = None

class Episode(BaseModel):
    episode_id: str = Field(default_factory=_uuid)
    trajectory_id: str
    start_step: int
    end_step: int
    sub_goal: str
    summary: str
    outcome: Literal["success", "partial", "failure"]
    key_steps: List[str] = []
    embedding: Optional[List[float]] = None

class Pattern(BaseModel):
    pattern_id: str = Field(default_factory=_uuid)
    condition: str
    action_template: str
    expected_effect: str
    scope: List[str] = []
    support_episodes: List[str] = []
    refute_episodes: List[str] = []
    alpha: float = 1.0
    beta: float = 1.0
    evidence_count: int = 0
    last_updated: datetime = Field(default_factory=datetime.utcnow)
    version: int = 1
    embedding: Optional[List[float]] = None

    @property
    def confidence(self) -> float:
        return self.alpha / (self.alpha + self.beta)

class Principle(BaseModel):
    principle_id: str = Field(default_factory=_uuid)
    statement: str
    scope: str = "cross-task"
    supporting_patterns: List[str] = []
    contradiction_log: List[dict] = []
    alpha: float = 1.0
    beta: float = 1.0
    embedding: Optional[List[float]] = None

    @property
    def confidence(self) -> float:
        return self.alpha / (self.alpha + self.beta)
```

---

## 文档变更记录

| 版本 | 日期 | 变更 |
|---|---|---|
| v0.1 | 2026-04-27 | 初稿 (与导师/合作者讨论前的 baseline 版本) |
