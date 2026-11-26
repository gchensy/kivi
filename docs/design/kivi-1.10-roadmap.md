# kivi 1.10 开发计划

1.9.0 已于 2026-11-26 发布，标签在 964e27db。`kivi-1.8-roadmap.md` 的计划至此全部完成。同一天，用户要求分析现有架构，并规划 1.9 之后的版本（“分析现在的项目架构，规划 1.9 之后的版本”）。本稿依据以下材料：

1. 对 master（964e27db）的结构核对：模块规模、模块间的导入、公开名称、测试和文档（第 1 节）；
2. 教程和指南里使用者要自己写的胶水代码（第 2 节）；
3. 1.8 计划第 5、6 节中不排期和维持不做的事项，以及欧盟《人工智能法》的最新时间表；
4. 一次本地测试：在 Python 3.14 上运行全部测试。

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
| 集成 PR | 教程和指南改用新对象，参考页和入口页 |

### 5.1 会话 `scorecard-spec`

暂定 API：

```python
# kivi.scorecard
ScorecardSpec(binning, points, *, pdo, base_score, base_odds,
              calibration=None, scale=None, kivi_version=None)
# binning: 评分卡各特征的 BinningSpec（BinningSpec.select 的结果）
# points: points_table 的结果，可以先取整
# calibration: None、{"shift": Δ}（central_tendency_shift）或 {"slope": a, "intercept": b}（platt_fit）
# scale: master_scale 的结果

card.points(frame)        # DataFrame：每个特征一列
card.score(frame)         # Series：逐列相加，相同的行分数相同
card.probability(frame)   # score_to_pd，再按 calibration 平移或缩放 log odds
card.grade(frame)         # 有 scale 时按 assign_grades 分级
card.reason_codes(frame)  # 与 scorecard.reason_codes 相同
card.to_json() / ScorecardSpec.from_json(text)
```

设计要点（会话在实现前确认）：

- **存分数表，不存系数：**
  - 分数表就是部署出去的评分卡，可以取整；`scorecard.rst` 已经说明取整的做法。
  - 取整后，PD 由取整后的分数换算，与生产系统一致。
  - 未取整的分数表可以由 WOE 反推出系数，所以不必单独存系数。
- **不嵌入 `FeatureSpec`：** 特征从交易等其他表计算，生产中在评分之前完成。两份 JSON 并列存放，指南说明两者怎么配合。
- **格式：**
  - `format` 为 `kivi.ScorecardSpec`，`format_version` 为 1，`BinningSpec` 的字典原样嵌入，带它自己的格式版本。
  - 与 `FeatureSpec` 一样拒绝重复的键，比较相等用 JSON 文本，并在 `tests/data/` 存一份固定文件，保证以后的版本能读。
- **未见组：** 沿用 `BinningSpec` 的 `unseen` 设置，未见组的分数按其 WOE 换算。规则与 `reason_codes` 的 `unseen_points` 一致。
- **文件：** `src/kivi/scorecard.py`、`tests/test_scorecard.py`、`docs/user_guide/scorecard.rst`、`docs/api/scorecard.rst` 和一个片段。
  - 不改 `spec.py`，只用 `BinningSpec` 的公开方法。
  - `FeatureSpec` 的 JSON 辅助函数是 `features.py` 的私有函数（`_unique_keys`）。如果需要同样的函数，移到 `_validation`，并在 PR 说明里写明。

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

### 5.4 发布

1.10.0 等用户的指示。

## 6. 1.10 之后

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

新增两项：

| 内容 | 不排期的理由 |
| --- | --- |
| 公平性检查 | 见第 6 节 |
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
| 2 | `ScorecardSpec` 的范围 | 待定 | 存分箱和分数表，校准和主标尺可选，不嵌入 `FeatureSpec`，逐列求和 |
| 3 | 转移矩阵单因子模型 | 待定 | 采用 Belkin–Suchower–Forest 模型，先核对原文，再审定 API；与 `scorecard-spec` 并行，或推迟到以后 |
| 4 | 最新 Python 的 CI 任务 | 2026-11-26 已定，#114 已合并 | 增加 `newest-python` 任务，不设为必需检查 |
| 5 | 精简 CLAUDE.md | 2026-11-26 已定，#114 已合并 | 精简到 40.0 KB（目标约 30 KB），算法细节留在 docstring、设计稿和测试里 |
| 6 | 大模块 | 待定 | 不主动拆 |
| 7 | 最低版本 | 待定 | Python 3.11 停止支持（2027 年 10 月）后提高到 3.12，由前一个版本预告；NumPy 和 pandas 不动 |
| 8 | 公平性检查 | 待定 | 不排期，2027 年年中再评估 |
| 9 | 暂定名称 | 待定 | `ScorecardSpec`、`shift_matrix`、`matrix_factor`，由用户在 PR 审阅时确认 |

## 10. 协调

| 会话 | 拥有的文件 |
| --- | --- |
| `scorecard-spec` | `src/kivi/scorecard.py`、`tests/test_scorecard.py`、`tests/data/scorecard_spec_*.json`、`docs/user_guide/scorecard.rst`、`docs/api/scorecard.rst`；如需移动 `_unique_keys`，还有 `_validation.py` 和 `features.py` |
| `matrix-scenarios` | `src/kivi/lifetime.py`、`tests/test_lifetime.py`、`docs/user_guide/lifetime.rst`、`docs/api/lifetime.rst`、`docs/design/kivi-1.10-matrix-scenarios.md` |
| 协调会话 | 本稿；第 1 步的 PR（`.github/workflows/docs.yml`、`CLAUDE.md`）；集成 PR；发布 |

- **共享文件：** `kivi.po` 是唯一的共享文件，冲突用 `--merge-po` 解决。
- **合并顺序：** 第 1 步的 PR 合并之后，再开 1.10 的会话；功能 PR 都合并之后，再开集成 PR。
