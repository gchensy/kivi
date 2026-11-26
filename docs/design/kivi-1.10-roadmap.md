# kivi 1.10 开发计划

1.9.0 已于 2026-11-26 发布，标签在 964e27db。`kivi-1.8-roadmap.md` 的计划至此全部完成。同一天，用户要求分析现有架构，并规划 1.9 之后的版本（“分析现在的项目架构，规划 1.9 之后的版本”）。本稿依据以下材料：

1. 对 master（964e27db）的结构核对：模块规模、模块间的导入、公开名称、测试和文档（第 1 节）；
2. 教程和指南里使用者要自己写的胶水代码（第 2 节）；
3. 1.8 计划第 5、6 节中不排期和维持不做的事项，以及欧盟《人工智能法》的最新时间表；
4. 一次本地测试：在 Python 3.14 上运行全部测试；
5. 同日的另一份草稿：分支 `design/post-1.9-improvements` 上未提交的 `kivi-1.10-scorecard-spec.md` 和 `kivi-1.11-workflow-and-perf.md`。用户要求不改那份草稿，“只吸取可以执行的部分即可，纳入你后续开发的范围”；吸收的内容在第 5.1、5.4 和 6.1 节，没有采纳的写在第 6.1 节和第 7 节，并附理由。

第 3 节按优先级排定顺序：先做一个基础设施 PR（第 4 节），再做 1.10（第 5 节）。本稿不新增依赖。流程沿用 1.8：

- 开工前审定 API，需要依据文献的部分先核对原文；
- 每个功能会话拥有一组文件，与其他会话不重叠；入口页由集成 PR 统一改写；
- 合并和发布都等用户的指示。

## 1. 现有架构

### 1.1 规模

| 项目 | 数量 |
| --- | --- |
| 公开模块 | 22 个，另有 `kivi.datasets` |
| 公开名称（各模块 `__all__` 之和） | 130 个 |
| 源代码 | 28,125 行 |
| 测试 | 27,915 行，5,123 个测试，分支覆盖率 99.59% |
| 文档 | 24 篇指南、11 章教程、24 个 API 页 |
| 中文目录 `kivi.po` | 6,158 条，2.27 MB |

最大的几个模块：

| 模块 | 行数 | 公开名称 |
| --- | --- | --- |
| `calibration` | 2,759 | 20 |
| `binning` | 2,203 | 11 |
| `features` | 2,071 | 3 |
| `spec` | 1,946 | 2 |
| `plotting` | 1,899 | 16 |
| `vintage` | 1,785 | 7 |
| `lifetime` | 1,615 | 7 |
| `metrics` | 1,475 | 10 |

### 1.2 分层

- **私有层：** `_validation`（37 个函数）、`_counts`、`_edges`、`_special`、`_stats`、`_logit` 和 `_deprecation`。它们只被其他模块导入，自己不导入任何公开模块。
- **公开模块之间只有五处依赖：**
  - `monitoring` → `metrics`
  - `selection` → `metrics`
  - `rules` → `binning`
  - `spec` → `binning`、`woe`
  - `univariate` → `binning`、`metrics`、`woe`
- **没有循环导入，也没有公开模块调用另一个公开模块的私有函数。** `tests/test_namespaces.py` 检查每个公开模块只绑定 `__all__`、私有名称和模块。

结论：分层清晰，不需要重构。

### 1.3 拟合与应用

kivi 的约定是：在开发样本上拟合一次，再把结果原样用于所有样本。现有的配对如下：

| 拟合 | 应用 | 能否保存为带版本号的 JSON |
| --- | --- | --- |
| `bin_edges` 及有监督的 `*_edges` | `cut` | 经 `BinningSpec` |
| `category_groups`、`chimerge_groups` | `group_categories` | 经 `BinningSpec` |
| `woe_table` | `woe_transform` | 经 `BinningSpec` |
| `fit_spec` | `BinningSpec.woe_transform` | 能（格式版本 1、2） |
| `FeatureSpec`（只有定义，不拟合） | `FeatureSpec.compute` | 能（格式版本 1） |
| `logistic_regression` | 使用者自己写线性预测 | 不能 |
| `score_scaling`、`points_table` | `woe_to_points`、`pd_to_score`、`score_to_pd` | 不能 |
| `platt_fit`、`central_tendency_shift`、`quasi_moment_fit` | `platt_predict`，或使用者自己写平移公式 | 不能 |
| `master_scale` | `assign_grades` | 不能 |

### 1.4 测试与 CI

- **`build`：** 在 Python 3.12 上运行 ruff、mypy、`pytest --cov`、doctest 和两种语言的文档构建。
- **`minimum-versions`：** 在 Python 3.11 上，用 NumPy 1.24.4 和 pandas 2.0.3 运行 pytest。
- **最新的 Python 不在 CI 里。** Python 3.15.0 已于 2026-10-09 发布，NumPy 2.5.5、pandas 3.1.1、SciPy、scikit-learn、statsmodels 和 matplotlib 都有 cp315 的 wheel。2026-11-26 在本地用 Python 3.14.4、NumPy 2.5.5 和 pandas 3.1.1 运行全部测试：5,122 个通过，1 个跳过（mypy 检查，环境里没装 mypy）。3.15 没有在本地测成：本机 uv 0.11.26 只提供 3.15.0b3，为正式版构建的 NumPy wheel 在它上面一导入就段错误，这是 beta 与正式版的二进制接口不同，不能说明 kivi 的问题。3.15 的结果后来由第 1 步的 CI 任务给出（第 4 节）。`compatibility.rst` 写的是测试“on these oldest versions and on the newest releases”：NumPy 和 pandas 确实测了最新版，Python 却没有。

### 1.5 协作成本

`CLAUDE.md` 有 94.6 KB，约 2.4 万 token，每个会话开始时都要读入。其中大半是逐个函数的算法细节，与 docstring 和设计稿重复。多个会话并行时，这部分成本按会话数成倍增加。

## 2. 问题

### 2.1 部署链在分箱之后断开

评分卡在生产中的链条是：（特征）→ 分箱与 WOE → 系数 → 分数 → PD（校准）→ 等级和拒绝原因。kivi 只能保存其中的分箱。

- **教程明确把余下部分留给使用者：** `tutorial/monitoring.rst` 写道“The JSON is stored with the coefficients and the scale”，存储由使用者负责。
- **评分代码要手写，而且重复：** 教程有 8 处手写 `score_pd`，分布在 model、scorecard、evaluation、calibration、strategy、reject_inference 各章，monitoring 一章两处；指南有 2 处（strategy、calibration）。每一处都用矩阵乘积 `@`。
- **矩阵乘积会拆开平局：** 按 CLAUDE.md 的记录，`X @ coef` 对完全相同的行可能给出末位不同的结果，取决于 CPU 和 BLAS，所以 `selection_path` 改为逐列求和。手写的评分代码因此可能把 kivi 自己的函数保持相等的分数拆开。

后果：

- 每个使用者都要重写这段胶水代码；
- 系数、刻度、校准和主标尺的 JSON 没有格式版本，读回时也没有检查；
- 生产分数无法与开发时的分数逐一比较是否相等。

### 2.2 经济情景只平移违约概率

`user_guide/lifetime.rst` 的 “Economic scenarios” 一节这样构造情景：

- 用 `capital.conditional_pd` 平移各状态的条件 PD；
- 三个情景各用一个因子，分别是 1、0 和 −1.5，人为选定，没有从历史数据估计；
- 每年用同一个因子；
- 非违约状态之间的迁移概率保持不变。

Belkin、Suchower 和 Forest（1998）的单参数模型用一个系统因子平移整个转移矩阵，因此升级和降级也随经济变化。该模型还能从每年观察到的矩阵反推当年的因子，宏观模型可以接着预测这些因子。IFRS 9 第 5.5.17 段要求的前瞻性信息和压力测试常用这一方法。

kivi 已有其中三块：`transition_matrix`（计数）、`transition_pd`（矩阵的幂）和 `conditional_pd`（单个 PD）。缺的是对整个矩阵的条件化，以及因子的估计。

### 2.3 CI 不测最新的 Python

见 1.4 节。每有新的 Python 发布，同样的缺口都会再出现一次。

### 2.4 CLAUDE.md 太大

见 1.5 节。

### 2.5 核对后不算问题的

- **大模块：**
  - `calibration` 有 2,759 行，包含四个主题：先验修正、校准拟合、检验和主标尺。
  - 拆成包、把内容放进私有子模块，不改 API，但有三个代价：git blame 断开，测试要改导入，而热点并不会减少，因为功能 PR 本来就按模块划分。
  - 建议不主动拆。等某次计划让同一个模块成为两个会话的热点时再拆。
- **兼容性代码：**
  - 为 NumPy < 2.2（区间哈希会警告）和 pandas 2.0（可空类型）写的绕行，只有十几处注释和写法。
  - 提高最低版本的收益仍然很小，维持 1.8 的决定 6。

## 3. 优先级

排序看四点：对使用者的影响、拖着不做的风险、成本，以及有没有时间窗口。用户于 2026-11-26 认可这一顺序（“按这个顺序改写，先开第 1 步的 PR”）。

| 优先级 | 事项 | 价值 | 不做的风险 | 成本 |
| --- | --- | --- | --- | --- |
| **P0** | 最新 Python 的 CI 任务 | 确认 kivi 在 Python 3.15 上能用 | 3.15 已发布 7 周还没测过，坏了的话所有 3.15 用户都装不上 | 很小 |
| **P0** | 精简 CLAUDE.md | 每个会话少读入约 1.6 万 token，也少了前后不一致的地方 | 不影响使用者，但每个会话都多付成本 | 中等，不改代码 |
| **P1** | `ScorecardSpec` | 补上部署链，覆盖所有评分卡项目；教程和指南删去 10 处手写评分 | 使用者各自手写评分，相同的行可能得到末位不同的分数 | 最大；格式一旦发布就要长期兼容 |
| **P2** | 转移矩阵单因子模型 | IFRS 9 前瞻性信息和压力测试 | 现有写法是合理的简化，不算错 | 中等，要先核对原文 |
| 不动 | 拆大模块 | — | — | 本来就是“不做” |
| 不动 | 最低 Python 3.12 | — | 2027 年 10 月才到期，2027 年年中预告即可 | — |
| 不动 | 公平性检查、MoC 指南 | — | 2027 年年中再评估，或等实际需求 | — |

理由：

- **CI 任务排第一，因为它本身就是一次检查。** 如果 3.15 有问题，先修复并发布 1.9.1，再开始新功能。
- **CLAUDE.md 现在精简最省事：**
  - 规则要求 CLAUDE.md 只在没有并行 PR 时由协调会话修改，现在正好没有；
  - 先精简，1.10 的两个会话和集成 PR 都读较小的版本，集成 PR 也直接按新结构增加条目。
- **两项 P0 不改使用者可见的行为：** 合并到 master 即可，不发版本。
- **`ScorecardSpec` 是 1.10 的核心：** 它覆盖 kivi 最主要的用法，从分箱到分数、PD、等级和拒绝原因。保存格式要长期兼容，所以要早开工、多留审阅时间。
- **转移矩阵情景是第一个可以推迟的：**
  - 价值取决于组合：评级制矩阵（对公、小微和内评等级）常用这种模型；零售的逾期分档矩阵更常把违约率直接对宏观变量回归；
  - 它与 `scorecard-spec` 的文件不重叠，并行做的协调成本很低；
  - 并行做还是推迟，由用户决定（第 9 节决定 3）。

## 4. 第 1 步：基础设施 PR

由协调会话完成，分支从 master 开出，不发版本。

**已完成：** 用户于 2026-11-26 同意合并（“Merge after CI”），#114 于同日合并，合并提交为 109a9da2。

- `newest-python` 在 Python 3.15.2 上，用 NumPy 2.5.5、pandas 3.1.1 和 matplotlib 3.11.2 运行：5,122 个测试通过，1 个跳过（mypy 检查）。不需要 1.9.1。
- CLAUDE.md 从 94.6 KB 减到 40.0 KB，比约 30 KB 的目标大，因为剩下的都是规则和模块图。
- 合并前，一个独立的审阅代理逐条比对了新旧文件。它找出 6 条丢失或记录薄弱的规则、7 处不准确的陈述，均已改正。

以下是当初的计划。

- **`newest-python` 任务：**
  - 写在 `.github/workflows/docs.yml` 里，在最新的 Python（现在是 3.15）上安装 `.[test,plot]`，即最新的 NumPy、pandas 和 matplotlib，然后运行 pytest；
  - 它与 `minimum-versions` 一样不是必需检查，`build` 的名字和设置不动；
  - 每有新的 Python 正式版发布，就更新这个任务的版本，CLAUDE.md 写明这一点；
  - 有了它，`compatibility.rst` 所说的“the newest releases”对 Python 也成立。
- **精简 CLAUDE.md，目标约 30 KB：**
  - 保留：命令、所有规则、协作流程，以及每个模块一段的架构图，写明细节在哪里；
  - 删去：逐个函数的算法细节，这些已经在 docstring、设计稿和测试里；
  - 合并前请一个独立的审阅代理逐条核对：旧文件里的每一条规则，新文件里要么还在，要么能在 docstring、`docs/development/` 或设计稿里找到。

## 5. 1.10

| 会话 | 内容 |
| --- | --- |
| `scorecard-spec` | 可保存的整张评分卡：分箱、分数表、可选的校准和主标尺 |
| `matrix-scenarios` | 转移矩阵的单因子模型：按因子平移整个矩阵，并从观察到的矩阵估计因子。与 `scorecard-spec` 并行，或推迟到以后 |
| 协调会话 | `vif_filter` 提速，不改 API（第 5.4 节） |
| 集成 PR | 教程和指南改用新对象，参考页和入口页 |

### 5.1 会话 `scorecard-spec`

本节吸收了另一份草稿 `kivi-1.10-scorecard-spec.md` 中可执行的部分（来源见开头第 5 项）。暂定 API：

```python
# kivi.scorecard
ScorecardSpec(binning, table, *, pdo, base_score, base_odds,
              unseen_points=None, calibration=None, scale=None, kivi_version=None)
# binning: 评分卡各特征的 BinningSpec（BinningSpec.select 的结果）
# table: points_table 的结果（索引 variable，列 bin、woe、points），可以先取整
# unseen_points: 未见组按数值 WOE 处理的列 → 未见组的分值
# calibration: None，或 {"kind": "shift", "shift": Δ}、
#              {"kind": "platt", "slope": a, "intercept": b}、
#              {"kind": "isotonic", "edges": [...], "probability": [...]}
# scale: master_scale 的结果

ScorecardSpec.from_model(binning, coefficients, intercept, *, pdo, base_score, base_odds,
                         decimals=None, calibration=None, scale=None)

card.points(frame)                             # DataFrame：每个特征一列
card.score(frame)                              # Series：逐列相加，相同的行分数相同
card.probability(frame)                        # 见下文的校准口径
card.grade(frame)                              # 按 assign_grades 分级；没有 scale 时报错
card.reason_codes(frame, *, top=3, atol=1e-4)  # 自动传入 unseen_points
card.to_json() / ScorecardSpec.from_json(text)
```

设计要点（会话在实现前确认）：

- **存分数表，不存系数：**
  - 分数表就是部署出去的评分卡，可以取整。`from_model` 的 `decimals` 与 pandas 的 `round()` 相同，逢五取偶，也就是 `scorecard.rst` 已说明的做法。
  - 取整后，PD 由取整后的分数换算，与生产系统一致。
  - 用户手工调过分值时，直接用构造器；标准流程用 `from_model`，它调用 `points_table`，并在需要时先 `binning.select`。
- **未见组的分值单独存（`unseen_points`）：** 取整后的分数表不能由 WOE 精确反推系数，所以未见组的分值不能在评分时现算。
  - `unseen` 为 `"raise"` 的列：未见组报错，不需要分值；
  - `{"group": label}` 的列：用该组的分值；
  - 数值 WOE 的列：必须在 `unseen_points` 里给出分值，缺少或多给都报错；
  - `from_model` 按 `woe_to_points` 的同一公式算出未取整的分值，再与分数表一起取整；
  - `card.reason_codes` 把同一组 `unseen_points` 传给 `reason_codes`。
- **字段叫 `table`，方法叫 `points(frame)`：** 草稿把两者都叫 `points`，同一个类里不能共存。
- **校准写明作用在什么量上。** 设 s 为分数。未校准的 log-odds 直接按 (offset − s) / factor 计算，不先算 PD 再取 logit，以免极端分数处 PD 舍入成 0 或 1。三种校准：
  - `shift`：σ(log-odds + Δ)，Δ 来自 `central_tendency_shift`；
  - `platt`：σ(a · log-odds + b)，(a, b) 来自对 log-odds 拟合的 `platt_fit` 或 `quasi_moment_fit`，教程的校准一章正是这样拟合的；
  - `isotonic`：`isotonic_predict(s, edges, probability)`，(edges, probability) 来自 `isotonic_fit(s, target, increasing=False)`，因为分数越高风险越低。
  - 草稿另有一种直接在分数上做的 Platt 校准。log-odds 是分数的线性函数，两者只差一次换算；两种都提供，正好会造成草稿想避免的误用，所以只保留 log-odds 上的一种。
- **不嵌入 `FeatureSpec`：** 特征从交易等其他表计算，生产中在评分之前完成。两份 JSON 并列存放，指南说明两者怎么配合。
- **格式：**
  - `format` 为 `kivi.ScorecardSpec`，`format_version` 为 1。顶层键固定为 `format`、`format_version`、`kivi_version`、`scaling`、`binning`、`points`、`unseen_points`、`calibration` 和 `scale`，未设置的写 `null`。
  - `binning` 原样嵌入 `BinningSpec.to_dict()`，带它自己的格式版本。
  - 分值按 `binning` 各列表格的行顺序存成列表，分组标签和 WOE 都不再重写。`table` 的 `bin` 和 `woe` 本来就必须与分箱的表格逐行相同，所以这样存不丢信息，也用不到 `spec.py` 的私有编码函数（公开模块不能导入另一个公开模块的私有函数），`spec.py` 不用改。
  - 与 `FeatureSpec` 一样拒绝重复的键：`_unique_keys` 移到 `_validation`。另新写一个辅助函数，拒绝 `NaN` 和 `Infinity` 字面量；无穷写成 `{"float": "inf"}`。
  - 比较相等用 JSON 文本。`tests/data/` 存一份固定文件和它的评分结果（`scorecard_spec_1.10.0.json`、`.scores.json`），覆盖数值列、特殊值、归组的类别列、缺失值、未见组、校准和主标尺，保证以后的版本能读。
- **测试：**
  - 未取整时，`points`、`score` 和 `probability` 与 `woe_to_points`、`pd_to_score` 和模型的概率一致，相对误差 1e-12；
  - 相同的行分数按位相等；
  - 取整时，分数等于取整后分数表的逐列和；
  - 三种校准分别与对应函数对拍；
  - 未见组的三种设置与 `BinningSpec.woe_transform` 和 `reason_codes` 一致；
  - 覆盖全部错误分支。
- **文件：** `src/kivi/scorecard.py`、`tests/test_scorecard.py`、`tests/test_typing.py`、`tests/data/scorecard_spec_1.10.0*.json`、`docs/user_guide/scorecard.rst`、`docs/api/scorecard.rst` 和一个片段；移动 `_unique_keys` 时还有 `_validation.py` 和 `features.py`。
  - 不改 `spec.py`；`BinningSpec.from_json` 也不加重复键检查。

### 5.2 会话 `matrix-scenarios`

如果用户决定推迟，本节原样留到以后的版本。

先在 `docs/design/kivi-1.10-matrix-scenarios.md` 核对原文：

- Belkin、Suchower 和 Forest，“A one-parameter representation of credit risk and transition matrices”，*CreditMetrics Monitor*，1998 年第三季度；
- 至少一篇使用或检验该方法的文献。

确认以下几点：

- 阈值如何从每一行的累计概率得到；
- 因子用什么权重做最小二乘估计；
- 相关系数如何选取。原文的做法是让估计出的各期因子方差为 1。

API 经用户批准后再写代码，与 1.8 的 `default-rates` 相同。暂定 API：

```python
# kivi.lifetime
shift_matrix(matrix, *, correlation, factor, default_state=None)
# -> DataFrame，形状与 matrix 相同，各行之和为 1，违约行保持吸收
matrix_factor(counts, matrix, *, correlation, default_state=None)
# -> float：某一期观察到的计数矩阵相对于长期矩阵 matrix 的因子
```

约定和衔接：

- **因子的符号与 `capital.conditional_pd` 一致：** 正的因子表示经济好于平均。
- **衔接现有函数：** `shift_matrix` 的结果可以直接交给 `transition_pd`，每年用不同的因子时逐年相乘。
- **改写情景一节：** `lifetime.rst` 的 “Economic scenarios” 改用 `shift_matrix`，并增加一节，从 `make_loan_panel` 各年的矩阵估计因子。
- **文件：** `src/kivi/lifetime.py`、`tests/test_lifetime.py`、`docs/user_guide/lifetime.rst`、`docs/api/lifetime.rst`、设计稿和一个片段。

### 5.3 集成 PR

- **教程和指南改用 `ScorecardSpec`：**
  - 删去手写的 `score_pd`：教程 8 处，指南 2 处；
  - monitoring 一章保存 `ScorecardSpec`，不再分开存分箱和系数；
  - 按 CLAUDE.md 的规则重新运行受影响的页面，更新输出。
- **教程 portfolio 一章：** 是否改用 `shift_matrix` 构造情景，看会话的结果再定。
- **兼容性与参考页：**
  - `compatibility.rst` 增加 `ScorecardSpec` 格式和新表格的列；
  - `conventions.rst` 和 `glossary.rst` 收入新术语；
  - 两种语言的 README、`docs/index.rst` 和 `user_guide/index.rst` 的卡片。
- **CLAUDE.md：** 按第 4 节精简后的结构写入新模块的条目。

### 5.4 `vif_filter` 提速

协调会话的一个小 PR，来自草稿 `kivi-1.11-workflow-and-perf.md`。

- **现状：** 每轮淘汰都对 n × k 的标准化矩阵重做一次 SVD。2026-11-26 在本机测得，50,000 行 × 40 列、删去 10 列需要 833 ms；草稿在另一台机器上测得 572 ms。
- **做法：**
  - 循环前对全部标准化列做一次薄 QR 分解 Z = QR，之后每轮只对 R 的列子集做 SVD。
  - 正交变换不改变奇异值和右奇异向量，所以 VIF 在机器精度内不变。
  - 判定秩时仍用原来的行数。
- **不变的：** 公开 API 不变，平局规则（相对 1e-10）也不变。
- **测试：** `tests/test_selection.py` 加入与逐轮重算的等价性测试，覆盖满秩、近奇异、完全共线和行数少于列数四种情形。另加一个 `changed` 片段，写明提速。
- **随 1.10 发布：** 补丁版本只修错误，所以不出 1.9.1。
- **文件：** `src/kivi/selection.py`、`tests/test_selection.py`，与 1.10 的两个会话都不重叠。

### 5.5 发布

1.10.0 等用户的指示。

## 6. 1.10 之后

### 6.1 1.11 的候选

以下候选来自草稿 `kivi-1.11-workflow-and-perf.md`，都是新增函数、新增方法或可选关键字，默认结果不变。开工前由用户审定 API。

| 候选 | 解决的问题 | 文件 |
| --- | --- | --- |
| `rules.apply_rules(frame, candidates, *, special=None)` | 由 `rule_candidates` 的表生成每条规则的布尔列，代替手写 `((x > t) & ~x.isin(codes)).fillna(False)`；缺失值和特殊值不命中。在开发样本上，`rule_table` 的计数与候选表逐行相同 | `rules.py` |
| `rule_candidates(..., include_missing=False)` | 缺失本身风险高时，检出“缺失”规则；默认关闭，1.8 的结果不变 | `rules.py` |
| `BinningSpec.summary()` | 一张表看所有列：IV、箱数、WOE 趋势、最小的箱占比和事件数、缺失和特殊值的占比 | `spec.py` |
| `BinningSpec.replace_column(column, frame, target, *, edges=None, groups=None, ...)` | 人工调整一列的切点或归组，用开发样本重算该列的 WOE，其余列不动 | `spec.py` |
| `stepwise(..., include=None, max_variables=None)` | 业务上必须保留的变量，以及入模变量数的上限 | `selection.py` |

安排：

- `spec.py` 是热点文件，`summary` 和 `replace_column` 放在同一个会话里。
- `apply_rules` 和 `include_missing` 放在一个 `rules` 会话里。
- `stepwise` 的改动与第 5.4 节的提速是同一个文件，排在提速之后。

需要用户先决定的：

- **`stepwise(criterion="aic" | "bic")`：** 会重新打开 1.4 的决定。当时按指标选变量的需求由 `selection_path` 承担。
- **评分卡的 SQL 导出：**
  - CLAUDE.md 记录了用户的决定：不做 Spark 和 SQL 集成。
  - 生成 SQL 文本与访问数据不是一回事，可以单独论证，但要由用户决定。
  - 如果做，只生成分组、WOE 和分数的 `CASE` 表达式。PD 要用到 `exp`，各数据库的浮点和 `NULL` 语义不同，只用 sqlite3 测试不能保证与 Python 一致。
- **按页面拆分 `kivi.po`：**
  - 好处是消除并行 PR 在目录上的冲突。
  - 代价是 `build.py` 的检查、`--update-po`、`--merge-po`、CLAUDE.md 和 `documentation.rst` 都要跟着改。
  - 1.8 的冲突都由 `--merge-po` 自动解决了，等长期有三个以上并行会话时再评估。

### 6.2 最低版本与公平性检查

- **最低 Python 3.12：**
  - Python 3.11 的支持期到 2027 年 10 月结束（PEP 664）。
  - 按 `releasing.rst`，在那之前的最后一个版本用 `changed` 片段预告，那之后的第一个次版本生效。
  - NumPy 和 pandas 的最低版本不动，理由同 1.8 的决定 6。
- **公平性检查：**
  - 《人工智能法》附件三把自然人的信用评估和信用评分列为高风险用途。数字综合法案（Digital Omnibus on AI）于 2026 年 6 月通过，把附件三的义务推迟到 2027 年 12 月 2 日生效。
  - 委员会 2025 年 2 月关于“AI 系统”定义的指南不具约束力。它把以线性或 logistic 回归改进数学优化的系统作为例子，排除在定义之外，因此传统评分卡可能不受这些义务约束；这一点仍有争议。
  - 按受保护群体比较批准率、不利影响比率、AUC 和校准的表格，大部分是对现有函数的分组计算。
  - 建议现在不排期，2027 年年中再评估：看使用者是否在欧盟运行基于机器学习的信用模型，或被要求做偏差检查。

## 7. 不排期的内容

1.8 计划第 5 节的事项仍不排期，理由不变：

- 二维分箱、Cox 模型、beta 回归、下行 LGD、多期 Pluto–Tasche；
- 多期检验、带相关的二项检验和红绿灯法；
- 多变量规则树：`rule_candidates` 1.8.0 才发布，还没有使用反馈。

新增的几项：

| 内容 | 不排期的理由 |
| --- | --- |
| 公平性检查 | 见第 6.2 节 |
| L2 正则化的 `logistic_regression`（草稿） | 以平滑后的 WOE 为输入，很少出现完全分离。加了惩罚，p 值就不再是 Wald 检验，与 `stepwise` 的判据冲突。草稿把惩罚记在 `table.attrs` 里，这是 kivi 里的新做法，兼容性承诺不覆盖。真需要处理分离时，信用风险实务更常用 Firth 偏差修正，到时再立项 |
| 拆长函数（草稿：`rule_candidates`、`delinquency_panel`、`default_rate_table`、`fit_spec`） | 只改结构、不改行为，而这些文件正是 1.11 各会话要改的。只在已经拥有该文件的功能 PR 里顺带做 |
| `_best_binning` 的 `trend="any"` 快速分支（草稿） | `optimal_edges` 用 20 到 300 个初始箱也只要 7 到 116 ms；另写一条路径要完全复现现有的平局规则，风险大于收益 |
| `chimerge_groups` 的堆实现（草稿） | 3,000 个类别只要 125 ms，等有上万个类别的实际需求再做 |
| 保守性调整（MoC）的量化 | EBA/GL/2017/16 第 4.4.4 节把量化方法留给机构自己决定。其中统计部分（一般估计误差）已有 `roc_auc_interval`、`jeffreys_test` 和 `most_prudent_pd` 等区间可用，以后可以写成指南的一节 |

## 8. 维持不做

1.8 计划第 6 节不变：

- 市场风险、组合信用模型、经济资本、定价和 RAROC；
- 机器学习模型、图网络反欺诈和 AML；
- K-means/Jenks 分箱，CAIM、Ameva 和 Khiops；
- 多分类 WOE，折外 WOE 作为最终编码；
- Polars 等其他数据框输入；
- 输出下限。

## 9. 决定

| # | 问题 | 状态 | 建议 |
| --- | --- | --- | --- |
| 1 | 顺序 | 2026-11-26 已定（“按这个顺序改写，先开第 1 步的 PR”） | 第 3 节：先做第 1 步的基础设施 PR，再做 1.10；之后只按时间表提高最低 Python 版本 |
| 2 | `ScorecardSpec` 的范围 | 待定 | 第 5.1 节：存分箱、分数表和未见组的分值；校准三种（`shift`、`platt`、`isotonic`）和主标尺可选；不嵌入 `FeatureSpec`；逐列求和；`from_model` 一步生成 |
| 3 | 转移矩阵单因子模型 | 待定 | 采用 Belkin–Suchower–Forest 模型，先核对原文，再审定 API；与 `scorecard-spec` 并行，或推迟到以后 |
| 4 | 最新 Python 的 CI 任务 | 2026-11-26 已定，#114 已合并 | 增加 `newest-python` 任务，不设为必需检查 |
| 5 | 精简 CLAUDE.md | 2026-11-26 已定，#114 已合并 | 精简到 40.0 KB（目标约 30 KB），算法细节留在 docstring、设计稿和测试里 |
| 6 | 大模块 | 待定 | 不主动拆 |
| 7 | 最低版本 | 待定 | Python 3.11 停止支持（2027 年 10 月）后提高到 3.12，由前一个版本预告；NumPy 和 pandas 不动 |
| 8 | 公平性检查 | 待定 | 不排期，2027 年年中再评估 |
| 9 | 暂定名称 | 待定 | `ScorecardSpec`（字段 `table`、`unseen_points`，构造器 `from_model`）、`shift_matrix`、`matrix_factor`，由用户在 PR 审阅时确认 |
| 10 | 草稿中可执行的部分 | 2026-11-26 已纳入（“只吸取可以执行的部分即可，纳入你后续开发的范围”） | 第 5.1 节的 `ScorecardSpec` 细节和第 5.4 节的 `vif_filter` 提速归入 1.10；第 6.1 节的五项列为 1.11 候选，API 开工前审定 |
| 11 | `stepwise` 的 AIC/BIC 准则 | 待定 | 不做，除非有具体需求；按指标选变量已由 `selection_path` 承担 |
| 12 | 评分卡的 SQL 导出 | 待定 | 维持用户此前“不做 SQL 集成”的决定；如果要做，只生成分组、WOE 和分数的 `CASE` 表达式 |
| 13 | 拆分 `kivi.po` | 待定 | 维持单文件和 `--merge-po`，等长期有三个以上并行会话时再评估 |

## 10. 协调

| 会话 | 拥有的文件 |
| --- | --- |
| `scorecard-spec` | `src/kivi/scorecard.py`、`tests/test_scorecard.py`、`tests/test_typing.py`、`tests/data/scorecard_spec_*.json`、`docs/user_guide/scorecard.rst`、`docs/api/scorecard.rst`；如需移动 `_unique_keys`，还有 `_validation.py` 和 `features.py` |
| `matrix-scenarios` | `src/kivi/lifetime.py`、`tests/test_lifetime.py`、`docs/user_guide/lifetime.rst`、`docs/api/lifetime.rst`、`docs/design/kivi-1.10-matrix-scenarios.md` |
| 协调会话 | 本稿；第 1 步的 PR（`.github/workflows/docs.yml`、`CLAUDE.md`）；`vif_filter` 提速（`src/kivi/selection.py`、`tests/test_selection.py`）；集成 PR；发布 |

- **共享文件：** `kivi.po` 是唯一的共享文件，冲突用 `--merge-po` 解决。
- **合并顺序：** 第 1 步的 PR 合并之后，再开 1.10 的会话；功能 PR 都合并之后，再开集成 PR。
