# kivi 1.7 实施计划：策略规则与收益

2026-11-25 起草 · 待审定：第 6 节的 6 项

## 1. 目标与范围

1.7 的内容已由 `docs/design/kivi-1.4-risk-parameters.md` 第 7 节设计，2026-11-23 审定。该稿第 10 节第 14 项决定规则分析放进新模块 `kivi.rules`，第 15 项决定定价和 RAROC 只给指南示例。内容如下：

- **新模块 `kivi.rules`：**
  - `rule_table`：每条规则的命中与事件；
  - `rule_set_table`：规则集按顺序执行的瀑布表。
- **`kivi.strategy.cutoff_table`：** 增加 `revenue`，给出批准业务的收益和利润。
- **`kivi.metrics.expected_maximum_profit`：** 信用评分的期望最大利润（EMP）。

与 1.5、1.6 一样，开工前先核对原文，再审定 API。规则分析和收益列没有监管原文可依，核对的重点是 EMP 的定义和两个参考实现。本稿做三件事：

1. 记下核对结果和出处（第 2 节）；
2. 据此修订 API（第 3 节）；
3. 给出测试依据、会话分工和待你决定的事项（第 4–6 节）。

1.7 不需要新的运行时依赖：EMP 在 ROC 凸包上求和，只用 NumPy。

## 2. 原文核对结果

2026-11-25 核对，依据的文本和实现如下：

- **论文：** T. Verbraken, C. Bravo, R. Weber, B. Baesens, "Development and application of consumer credit scoring models using profit-based classification measures", *European Journal of Operational Research* 238(2), 2014, 505–513，doi:10.1016/j.ejor.2014.04.001。依据的是 KU Leuven Lirias 上的作者接受稿。草稿只写了"Verbraken 等"，第二、三作者是 Bravo 和 Weber。
- **R 包 `EMP`：** 2.0.6，2025-05-08，在 CRAN 上，未归档，依赖 ROCR。函数为 `empCreditScoring(scores, classes, p0 = 0.55, p1 = 0.1, ROI = 0.2644)`。
- **Python 包 `empulse`：** 0.14.0，2026-10-29。函数为 `empulse.metrics.empcs_score(y_true, y_score, *, success_rate=0.55, default_rate=0.1, roi=0.2644)`，拒绝比例用 `empcs_score.optimal_rate`。

### 2.1 EMP 的定义

| 项目 | 原文与实现 | 对 API 的影响 |
| --- | --- | --- |
| 类别与方向 | 论文以违约者为类别 0，分数越低越危险。R 包经 ROCR 以较大的标签为正类，`classes = 1` 为违约，分数即 PD，分数越高越危险。empulse 也是 1 为违约、分数越高越危险 | 与 kivi 的约定一致：分数随事件概率上升，`target` 为 1 表示违约 |
| 利润 | 每位申请人的平均利润 P(t; λ, ROI) = λ·π_d·F_d(t) − ROI·π_g·F_g(t)（论文式 1、12），以贷款金额为单位，按两类的先验比例加权。拒绝违约者的收益是 λ = LGD·EAD/A，误拒好客户的成本是 ROI | 结果是每位申请人的利润，不再归一化 |
| λ 的分布 | 在 λ = 0 处有点质量 p0（违约后全部收回），在 λ = 1 处有点质量 p1（全部损失），其余 1 − p0 − p1 均匀分布在 (0, 1) 上（§3.2） | 三个参数都做成关键字（决定 1 定名称） |
| 凸包上的计算 | 凸包顶点处 λ 的分界为 λ_{i+1} = π_g·ΔF_g·ROI / (π_d·ΔF_d)（式 13），对每段积分（式 15）。R 包分数相同的样本合为一个 ROC 点，凸包在其间插值 | 实现按 R 包的算法，见第 3.3 节 |
| 式 15 的笔误 | 论文在 p1 项用第 k+1 个顶点；λ = 1 时的最优点是第 k 个顶点，R 包和 empulse 都用第 k 个。逐点暴力求最大值证实后者正确：按论文原式，示例数据给 0.09417，正确值为 0.09747 | 按 R 包和 empulse；docstring 注明 |
| λ = 0 | 两个实现都在 λ = 0 时不拒绝任何人，这个点质量对利润的贡献为 0 | 同 |
| 拒绝比例 | η_emp = ∫ [π_d·F_d(T(λ)) + π_g·F_g(T(λ))] h(λ) dλ（式 6、8），即期望的拒绝比例。R 包的 `EMPCfrac` 和 empulse 的 `optimal_rate` 都是它。empulse 的 docstring 误写为"批准"的比例 | 结果中的 `rejected_share` |
| 默认值 | ROI = 0.2644 只出现在论文图 2 的示例中；p0 = 0.55 和 p1 = 0.1 是 R 包的默认值，接受稿中只定性描述了 λ 的分布，正式版是否给出未能核对 | 默认值取 R 包和 empulse 的值，docstring 说明出处 |
| 样本权重 | 两个实现都不接受权重（empulse 对 `sample_weight` 报 `TypeError`） | kivi 用频数权重计算 ROC 点，整数权重等于重复行 |

**参考值。** 用临时环境中的 empulse 0.14.0 算出：

| 数据 | `emp` | `rejected_share` |
| --- | --- | --- |
| `target = [0, 1, 0, 1, 0, 1, 0, 1]`，`score = [0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 0.8, 0.9]` | 0.0974701705 | 0.324345 |
| 完全分开（`score = target`，违约占一半） | 0.1375 | 0.225 |

**独立实现。** 另外用 NumPy 写了凸包算法，在 30 组随机样本上（其中 10 组有大量相同分数）与 empulse 的 `emp` 相差不超过 1.4e-17。

**empulse 能否做测试依赖。** 它要求 Python ≥ 3.11 和 NumPy ≥ 1.25.2，并依赖 sympy、narwhals 和 scikit-learn ≥ 1.5.2。kivi 的 `minimum-versions` 作业用 Python 3.10 和 NumPy 1.24.4，装不上它（决定 5）。

### 2.2 规则与收益

政策规则的命中率、提升度和瀑布表是业务惯例，没有监管原文。设计上沿用 kivi 已有的语义：

- **结果未知的申请。** 规则常对全部申请人评估，其中被拒的申请没有结果。`swap_set` 和 `monitoring_table` 把缺失的目标当作未知结果：计入总数，不计入已知和事件。
- **合成数据。** `make_applications` 已有两条政策规则：负债收入比上限和最低年龄，组成 `policy_decline`。它还给每个申请人一个真实的 `target`，可以直接用于指南。

## 3. API：在 1.4 稿第 7 节基础上的修订

```python
# kivi.rules（新模块）
rule_table(rules, target=None, *, weights=None)
rule_set_table(rules, target=None, *, weights=None)

# kivi.strategy
cutoff_table(score, target, *, cutoffs=10, loss=None, revenue=None, weights=None)

# kivi.metrics
expected_maximum_profit(score, target, *, no_loss_share=0.55, full_loss_share=0.1,
                        roi=0.2644, weights=None)
```

### 3.1 `kivi.rules`

- **`rules`：** DataFrame，每列一条规则，列名为规则名，值为布尔值或 0 与 1，命中（即拒绝或提示）为 True。缺失值报错（决定 4），因为规则在数据缺失时是否命中，是规则本身的定义。
- **`target`（决定 2）：** 可以不给。不给时只有命中列。给出时，缺失的标签是未知结果，与 `swap_set` 相同：计入 `hits`，不计入 `known` 和 `event`。
- **`rule_table`：** 索引为 `rule`，每条规则一行。
  - 命中列：`hits`、`hit_rate`、`unique_hits`。`unique_hits` 是只被这一条规则命中的样本。
  - 给出 `target` 时再加：
    - `known`：命中中结果已知的；
    - `event`；
    - `event_rate`：`event / known`；
    - `lift`：`event_rate` 除以全部已知结果的事件率；
    - `recall`：命中的事件占全部已知事件的比例；
    - `unique_event`：只被这一条规则命中的事件。
- **`rule_set_table`（决定 3）：** 按列的顺序依次执行，每行一条规则，每行只计入前面的规则都没有命中的样本。
  - 新命中列：`new_hits`，给出 `target` 时还有 `new_event`。
  - 累计列：执行到这一条规则为止仍然批准的业务。列名与 `cutoff_table` 相同，读起来像以规则为截断点的 `cutoff_table`：
    - `approved`、`approval_rate`；
    - 给出 `target` 时再加 `approved_event`、`approved_event_rate`、`rejected_event_share`、`rejected_non_event_share`，比例的分母是已知结果。
- **重叠矩阵：** 用 `rules.T @ rules`，指南给出示例，不另设函数。
- **`weights`：** 频数权重。

### 3.2 `cutoff_table` 的收益

- **`revenue`：** 每个样本的收益，例如利息收入或净息差，必须有限且非负（决定 6），按 `target` 的方式对应。
- **新列：** `approved_revenue`；与 `loss` 同时给出时，再加 `profit = approved_revenue − approved_loss`。两列都追加在已有的列之后，符合兼容性规则。
- **最优截断点：** 由 `table["profit"].idxmax()` 得到，不另设函数。

### 3.3 `expected_maximum_profit`

- **参数（决定 1）：** 改用描述性的名称：`no_loss_share` 是论文和 R 包的 p0，`full_loss_share` 是 p1，`roi` 是 ROI。这两个比例要在 [0, 1] 内，且和不超过 1；`roi` 要为正。
- **结果：** 以 `statistic` 为索引的 Series，含 `emp` 和 `rejected_share`。
- **算法**，与 R 包和 empulse 一致：
  1. 按分数从高到低累计违约者和好客户的（加权）数目，相同的分数合为一点；
  2. 取 ROC 点从 (0, 0) 到 (1, 1) 的上凸包；
  3. 算出各段的 λ 分界，取最后一个小于 1 的分界 k；
  4. 对均匀部分逐段积分，再加上 p1 在第 k 个顶点处的值；λ = 0 时不拒绝任何人。
- **规则同其他指标：**
  - 分数随违约概率上升，传 PD 即可；
  - 两类都要有；
  - 分数有限；
  - `weights` 是频数权重。

## 4. 测试与参考

| 内容 | 对照 |
| --- | --- |
| `expected_maximum_profit` | 第 2.1 节 empulse 的参考值，写死在测试里；完全分开时的闭式值；在 λ 网格上逐点暴力求最大利润再积分的独立实现；整数权重等于重复行；相同分数；只有一类时报错；三个参数的边界 |
| `cutoff_table` 的收益 | 手算；`profit` 等于两列之差；整数权重等于重复行；默认输出逐位不变 |
| `rule_table` | 手算；与 pandas 的 `groupby` 和 `rules.T @ rules` 比对；未知结果；整数权重等于重复行 |
| `rule_set_table` | 逐行重放规则的暴力实现；最后一行与对全部规则取或之后的 `cutoff_table` 式汇总一致；顺序改变时累计列的最后一行不变 |

## 5. 会话与流程

| 会话 | PR | 文件 |
| --- | --- | --- |
| `rules` | 一个 PR | `src/kivi/rules.py`（新）、`tests/test_rules.py`（新）、`docs/user_guide/rules.rst`（新）、`docs/api/rules.rst`（新）、`tests/test_namespaces.py`（公共模块数）、发布说明片段 |
| `profit` | 一个 PR，与 `rules` 并行 | `src/kivi/strategy.py`、`src/kivi/metrics.py`、`tests/test_strategy.py`、`tests/test_metrics.py`、`docs/user_guide/strategy.rst` 和 `metrics.rst` 的新小节、`docs/api/metrics.rst`、发布说明片段 |

两个会话的文件互不重叠，`kivi.po` 用 `--merge-po` 解决冲突。规则指南用 `make_applications` 的申请人，以负债收入比和年龄的政策规则为例，再加几条基于其他列的规则。

流程：

1. 本稿审定。
2. 两个会话开工。我逐个评审，按你的指示合并。
3. 集成 PR 由我来写：
   - README、首页和指南索引：`rules` 指南的位置由集成 PR 定，建议放在 `strategy` 之后；
   - `compatibility.rst`：
     - `rule_table`、`rule_set_table` 的列；
     - `cutoff_table` 的新列；
     - EMP 的 `statistic` 索引；
   - glossary 与 conventions：
     - `rules`、`revenue`、`roi` 等参数；
     - 政策规则、命中率、提升度、瀑布表、EMP、ROC 凸包等术语；
   - 测试登记表；
   - 图：规则瀑布、利润随截断点的变化；
   - CLAUDE.md。
4. 发布 1.7.0，需要你的确认。

1.7 之后，1.4 稿的路线图就全部完成了。

## 6. 决定

以下 6 项待审定，推荐方案在前。

| # | 问题 | 推荐 | 其他选项 |
| --- | --- | --- | --- |
| 1 | EMP 的参数名 | `no_loss_share`、`full_loss_share`、`roi`，描述性，docstring 写明对应论文的 p0、p1 | 草稿和 R 包的 `p0`、`p1`、`roi`；empulse 的 `success_rate`、`default_rate`（后者与 kivi 中"违约率"的含义冲突） |
| 2 | 规则分析的目标 | 可以不给；缺失的标签是未知结果，与 `swap_set` 相同，规则可以对全部申请人评估 | 草稿：必须给出、只能是 0 与 1 |
| 3 | `rule_set_table` 的列 | 新命中列 `new_hits`、`new_event`，累计列与 `cutoff_table` 同名（`approved`、`approval_rate`、`approved_event_rate`、`rejected_event_share`……） | 草稿的 `new_hits`、`new_event`、`declined`、`recall`、`remaining_event_rate` |
| 4 | 规则中的缺失值 | 报错，由调用者先决定数据缺失时规则是否命中 | 记为未命中 |
| 5 | EMP 的测试参考 | 把 empulse 0.14.0 的参考值写死在测试里，另加逐点暴力求最大值的独立实现 | 把 empulse 加入测试依赖：要求 Python ≥ 3.11 和 NumPy ≥ 1.25.2，`minimum-versions` 作业装不上，需要另设 extra 并在该作业中跳过 |
| 6 | `revenue` 的取值 | 有限且非负，与 `loss` 相同，费用放进 `loss` | 允许负值 |
