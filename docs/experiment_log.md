# RM 实验日志 (Experiment Log)

> **文件用途**：每跑一次实验就**追加一节**，把"命令、随机种子、模型、数据、结果、观察、下一步"全部记下来。
> 不要靠记忆，不要等 W&B 说话——这份文件是**复现性的最后兜底**。
>
> 配套：`docs/coding_log.md` 记录"为什么代码是这样写"，本文件记录"代码运行得到了什么"。
>
> **填写守则**
> 1. 实验编号 = `E{阶段}.{序号}`，如 `E1.01`、`E1.02`、`E2.01`。
> 2. 失败的实验也要记，放最后的 "Failed runs" 区。
> 3. 关键结果（SR / Steps / Tokens / |M| / RL / BRA / SMP）以 markdown 表格记录，方便 grep。
> 4. 如果一个实验的产物（trace、checkpoint）值得保留，把路径写到 `Artifacts:` 字段。
> 5. 用绝对日期（YYYY-MM-DD），不要写"昨天"。

---

## 模板（每个实验复制一份）

```
### EX.YY — <一句话标题>

* **日期**：YYYY-MM-DD
* **关联代码**：commit `<hash>` / `coding_log.md` 第 X 阶段第 Y 轮
* **目标**：本次想验证什么假设 / 想拿到什么数？

**配置**
* env: alfworld / scienceworld / webshop / mock
* split: eval_out_of_distribution / ...
* agent: react / no_mem / rm / reflexion / mem0 / ...
* llm: qwen7b / qwen14b / gpt4o / deepseek / mock
* embedder: BGE-M3 / Qwen3-Embedding / mock
* seed: 42 (主) / 0 / 1 / 2 (其他种子)
* n_tasks: ...
* n_trials_per_task: ...
* 关键超参覆盖：`rm.surprise.tau_low=0.2`, ...
* 算力：1×A100 80GB / 4090 24GB / CPU only

**命令**
\```bash
python scripts/XX_run_xxx.py --xxx ...
\```

**结果**

| 指标 | 值 | 备注 |
|---|---|---|
| SR | ... | |
| Avg steps | ... | |
| Tokens / task | ... | |
| \|M\| (n_patterns) | ... | |
| Refutation Lag (RL) | ... | 仅 Refute 实验 |
| Belief Revision Acc (BRA) | ... | 仅 Refute 实验 |
| Stale-Memory Penalty (SMP) | ... | 仅 Refute 实验 |

**观察 / 解释**
* （3–5 条 bullet：模型在哪类任务上失败？memory 体积怎么变？token 成本是否 amortise？）

**Artifacts**
* W&B run: https://wandb.ai/...
* trace dump: `runs/<timestamp>_<exp>/trajectories.jsonl`
* memory dump: `runs/<timestamp>_<exp>/memory.sqlite`

**Follow-ups**
* （列出本次跑出来后，下次该补充什么实验或修哪个 bug）
```

---

## 第 1 阶段实验

> _本节由用户在跑实验后填充。_

### E1.00 — 准备实验环境（占位）

* **日期**：（待填）
* **关联代码**：阶段 1 第 1 轮编码完成后
* **目标**：验证 self-check / 单元测试可重复跑通；记录基础 dev 环境快照（OS、Python、GPU、关键包版本）。

**配置**
* env: 不适用
* agent: 不适用
* llm: mock
* （记录：`conda env export -n rm > runs/snap/env_E1.00.yml`）

**命令**
```bash
conda activate rm
pip install -e ".[dev]"
python scripts/99_self_check.py
pytest -q
ruff check src/ tests/ scripts/
```

**预期结果**
| 指标 | 值 |
|---|---|
| Self-check pass | 7 / 7 |
| pytest pass | 43 / 43 |
| ruff errors | 0 |

**实测结果**：（待填）

**观察 / 解释**：（待填）

---

### E1.01 — RM 端到端 mock 烟测（占位）

* **日期**：（待填）
* **关联代码**：阶段 1 第 2 轮编码完成后
* **目标**：验证 ReflectiveAgent + Writer + Updater + Retriever 全链路在 mock 环境下能产出 Pattern/Principle/Update 三类记忆；建立后续真 LLM / 真 env 跑实验的对照基线。

**配置**
* env: mock（goal_keyword="GOAL"）
* agent: rm（ReflectiveAgent）
* llm: mock（content-routed JSON 回复）
* embedder: mock（dim=64）
* surprise_backend: embed_delta（同时跑一次 llm_judge 做对比）
* n_tasks: 8
* max_steps: 5
* reflect_every_n_trajectories: 4

**命令**
```bash
# 默认 backend=embed_delta
python scripts/02_run_rm_mock.py --n_tasks 8 --max_steps 5

# 切 backend=llm_judge 对比
python scripts/02_run_rm_mock.py --n_tasks 8 --max_steps 5 --surprise_backend llm_judge

# 同时再跑一次单测和 lint，确保实验前代码状态干净
pytest -q
ruff check src/ tests/ scripts/
```

**预期结果**
| 指标 | 值 |
|---|---|
| pytest pass | 81 / 81 |
| ruff errors | 0 |
| SR (8 任务) | 8 / 8 = 100 % |
| Avg steps | 1.00 |
| `|M|.patterns` | ≥ 1 |
| `|M|.principles` | ≥ 1 |
| `|M|.updates` | ≥ 1 |
| Tokens (mock) | 几百级别 |

**实测结果**：（待填）

**观察 / 解释**：（待填）

**Follow-ups**：
* 如果 patterns 计数始终是 1（被反复 merge），说明 mock 环境过于单一——下一步用真 LLM + ALFWorld 验证 Pattern 多样性。
* 切到 `llm_judge` 后 surprise 分布是否变化？记一张直方图。

---

<!-- 后续每跑一次实验在此处追加 ### EX.YY — <标题> 一节 -->

## Failed runs（汇总区）

> 失败的、被废弃的、或被新版本取代的实验放这里。保留命令与结论的核心 1–2 句即可。
> 失败也是论文的素材——尤其能说明哪些方案不 work。

| 实验编号 | 日期 | 一句话结论 | 失败原因 |
|---|---|---|---|
| (示例) E2.07 | 2026-06-XX | RM w/o Bayesian (硬计数) 比 Beta 后验差 4 pp | hard-count 对噪声不鲁棒 |
| | | | |
