# kivi 1.4 设计稿：补齐 0.2.x 的遗留功能

2026-11-23 起草 · 2026-11-23 审定：最后一节的 10 项全部采用推荐 · 2026-11-24 按合并的实现更正 `vif_filter` 的并列规则：`priority` 也相同时剔除列顺序中最后的那个

## 目标与范围

1.4 有两份设计稿，分别审定：

- **本稿**：补齐 0.2.x 中既没有迁移、迁移指南也没有说明的功能。由协调会话起草。
- **领域稿**：新的风控算法。由“金融风控算法研究”会话起草，即 PR #79 的 `docs/design/kivi-1.4-risk-parameters.md`。按它的推荐，1.4 只做验证报告的检验，资本计量、生存曲线、LGD 和规则放到 1.5–1.7。

两份稿子涉及的源文件互不重叠，见“与领域稿的协同”一节。

本稿的依据是 2026-11-23 对 `backup/kivi` 的逐项核对：94 个文件、169 个公开名称，每个都对照了 `docs/release/migration.rst` 和 kivi 的 API，没有对应的再读原代码。迁移指南已经说明不做的部分（Spark、SQL、模型包装、`KmeansBins`、Beta 分级等）维持原决定，本稿不重议。

**要做的：**

- **VIF 逐步剔除：** `kivi.selection.vif_filter`，对应 `VIF.LowVIFFeatures`。
- **按排序逐个加入变量的模型报告：** `kivi.selection.selection_path`，对应旧 `StepWise` 的实际做法。
- **两张图：** `kivi.plotting.distribution_chart` 画好坏客户在各分段的分布，对应 `BinaryMetrics.plot_score`；`kivi.plotting.precision_recall_chart` 画精确率–召回率曲线，对应 `pr_curve`/`plot_pr`。
- **WOE 趋势标签：** `kivi.woe.woe_trend`，对应 `woe/utils.monotony`。
- **迁移指南更正：** 把上面这些和不做的小工具写清楚，放在集成 PR。

运行时仍只依赖 NumPy 和 pandas，图仍只需要可选的 `kivi[plot]`。

**不做的**（只在迁移指南里说明）：

- `int_score`：把分数按 5 取整。用 pandas 的 `round` 即可，而且取整会让各项分数之和偏离总分。
- `StepWise.top_n_features`：每个维度取前 N 个变量。`dimension` 是变量到维度的映射时，用 `profile.assign(dimension=dimension).sort_values("iv").groupby("dimension").tail(n)` 即可。
- `plot_confusion_matrix`：`confusion_table` 每个截断点一行，已经是表格。
- `MasterScale.plot_grade_rate`：画的是 Beta 分级的结果，而 Beta 分级已决定不迁移。`calibration_chart` 能看各等级的预测与实际。

## 背景：核对结果

| 0.2.x | 当时的做法 | kivi 1.3 | 本稿 |
| --- | --- | --- | --- |
| `VIF.LowVIFFeatures(df, thresh=5)` | 每轮试着去掉每个变量，选出去掉后其余变量 VIF 之和最小的那个，直到最大 VIF 低于阈值；直接修改传入的 DataFrame | `vif` 只计算，剔除靠手工 | `vif_filter` |
| `StepWise(indicator="auc", early_stop=...)` | **不是**按显著性的逐步回归：先按单变量指标（IV、KS 等）给变量排序，然后依次拟合前 1 个、前 2 个……前 k 个变量的模型，记录每一步的 AUC、KS、lift 和系数符号；指标的相对增量低于 `early_stop` 时停止。`backward` 是同一组模型倒过来 | `stepwise` 按 p 值逐步选择，语义不同，迁移指南只写了“`stepwise` 替代 `StepWise`” | `selection_path` |
| `BinaryMetrics.plot_ks` | 按分数分桶画好坏客户的累计占比和 KS | **已覆盖**：`gains_chart` 画同样两条累计曲线，并标出 KS | 只更正迁移指南 |
| `BinaryMetrics.plot_score` | 好坏客户分数的直方图加核密度 | 无 | `distribution_chart` |
| `BinaryMetrics.pr_curve`、`plot_pr` | 精确率–召回率曲线 | `roc_table` 已有 `precision` 和 `true_positive_rate` 列，但没有图 | `precision_recall_chart` |
| `woe/utils.monotony` | 给一列数的趋势打标签：单调上升、单调下降、先升后降、先降后升或未知；相等算作上升 | 无；`monotonic_edges` 能保证单调，但不能判断现有分箱的形状 | `woe_trend` |
| `int_score`、`top_n_features`、`plot_confusion_matrix`、`plot_grade_rate` | 见上一节 | — | 不做，写进迁移指南 |

## API 提案

### `kivi.selection.vif_filter`

```python
vif_filter(X, *, threshold=5.0, priority=None) -> pd.Index
```

- 与 `correlation_filter` 同形：返回保留下来的列标签，按 `X` 的列顺序。`X` 不是 DataFrame 时返回列位置。
- 每轮用 `vif` 计算当前保留列的 VIF。最大值不超过 `threshold` 就停止，否则按决定 1 的规则去掉一个变量再算。只剩一列时停止，因为单列的 VIF 为 1。
- `priority` 与 `correlation_filter` 相同，每列一个数，通常传 IV。它的作用由决定 1 决定。
- 输入检查与 `vif` 相同。`threshold` 必须是大于 1 的有限实数。不修改 `X`。

### `kivi.selection.selection_path`

```python
selection_path(X, target, *, order=None, validation=None, weights=None) -> pd.DataFrame
```

- `order`：变量加入的顺序，是 `X` 的列标签序列（数组输入时是列位置），通常是 `profile_table` 或 `woe_table` 按 IV 从高到低的结果。可以只列一部分列。省略时按 `X` 的列顺序。
- 第 k 行是用 `order` 的前 k 个变量拟合的逻辑回归，拟合方法与 `logistic_regression` 相同。返回的表：
  - 索引名为 `step`，取值 1 到 k；
  - `variable`：这一步加入的变量；
  - `auc`、`gini`、`ks`：模型拟合概率在开发样本上的指标，与 `roc_auc`、`gini`、`ks` 的定义相同；
  - `max_p_value`：这一步模型中最大的 Wald p 值；
  - `min_coefficient`：最小的系数，不含截距。WOE 输入时它不大于 0，说明有变量的方向反了，即 `stepwise(positive=True)` 会拒绝的情况。
- `validation`：可选的 `(X_valid, target_valid)`，见决定 4。给出时多三列 `validation_auc`、`validation_gini`、`validation_ks`，用开发样本拟合的模型给验证样本打分后计算。
- `weights`：频数权重，语义同 `logistic_regression`，指标也加权。
- 不内置提前停止（决定 5）：表里有每一步的结果，增量用 `.diff()` 即可看到。

### `kivi.plotting.distribution_chart`

```python
distribution_chart(table, *, ax=None) -> Axes
```

- 对表的每一行，画该行占全部非事件的比例和占全部事件的比例，两组柱子并排，按表的行顺序排列。
- 只需要 `non_event` 和 `event` 两列，所以 `gains_table`（分数段）和 `woe_table`（分箱）都能直接画。占比按原始计数计算，不做平滑。
- 图中文字只用列名和数据值，与其他图一致。

### `kivi.plotting.precision_recall_chart`

```python
precision_recall_chart(table, *, label=None, ax=None) -> Axes
```

- 输入 `roc_table`。横轴 `true_positive_rate`（即召回率），纵轴 `precision`。
- 跳过阈值为 `inf` 的第一行：那一行没有人被标记，精确率没有定义。
- 画一条水平线表示事件率，即随机排序分数的精确率。`label`、`ax` 与 `roc_chart` 相同。

### `kivi.woe.woe_trend`

```python
woe_trend(table) -> str
```

- 输入 `woe_table` 的结果，或 `BinningSpec.tables` 里的一张表。
- 只看索引标签是区间的行，按表的顺序。特殊值和缺失值的行不参与判断。
- 返回下面之一：
  - `"increasing"`、`"decreasing"`：单调上升、下降；
  - `"u_shaped"`：先降后升；
  - `"inverted_u"`：先升后降；
  - `"irregular"`：方向变了不止一次；
  - `"flat"`：全部相等，或者只有一个区间行。
- 没有区间行时（类别变量）报 `ValueError`，因为无序的类别谈不上趋势。
- 判断用 `woe` 列。相邻两值的相对差在 1e-10 以内算相等，与其他统计量的比较一致；相等如何处理见决定 9。

## 语义细节

- **`vif_filter`：**
  - 无穷大的 VIF（完全共线）先剔除。多个变量同为无穷大时，按决定 1 的并列规则处理。
  - 结果是确定的：在 `priority` 也相同时，剔除按 `X` 的列顺序排在最后的那个，靠前的列得以保留，与 `correlation_filter` 一致。这样按 IV 排好序的 `X` 即使不传 `priority`，也会保留预测力更强的变量。
- **`selection_path`：**
  - 某一步的模型完全分离两类时，报 `SeparationError` 并在信息里写明是第几步，与 `logistic_regression` 一致。用户可以缩短 `order`。
  - 不收敛时的 `RuntimeWarning` 与 `stepwise` 相同。
  - 分数是拟合的违约概率，随风险上升，与 `kivi.metrics` 的约定一致。
- **两张图：**
  - 空的分数段保留（计数为 0 的柱子），与 `gains_table` 保留空段一致。
  - 输入检查和报错方式与现有的 9 张图相同。
- **`woe_trend`：** 标签字符串属于公开接口，写进 `compatibility.rst`。

## 验证与测试

- **`vif_filter`：**
  - 人工构造的共线数据：x3 = x1 + x2 + 噪声，以及完全共线；
  - `priority` 和并列的情况；
  - 每一轮的选择与逐次调用 `vif` 的结果一致；
  - 不修改输入。
  - 如果决定 1 选旧版规则，再与 `LowVIFFeatures` 在同一组数据上逐轮比对。
- **`selection_path`：**
  - 每一行与用同样列调用 `logistic_regression` 再算 `roc_auc`、`gini`、`ks` 的结果完全相同；
  - 整数权重等于重复行；
  - 在 bank 数据上与旧 `StepWise` 逐步比对 AUC：两边都是同一个逻辑回归的最大似然估计，预期差异在 1e-6 量级。
- **两张图：** 沿用 `tests/test_plotting.py` 的做法，检查画出的数据，不比对像素。
- **`woe_trend`：**
  - 表驱动测试：单调、U 形、倒 U、多次转向、相等、单行，以及带特殊值和缺失值的表；
  - 在随机序列上与旧 `monotony` 的标签逐一对应。旧版把相等算作上升，所以比对只用不含相等值的序列。

## 与领域稿的协同

- **本稿的功能 PR 独占：** `src/kivi/selection.py`、`src/kivi/plotting.py`、`src/kivi/woe.py`，以及它们的测试、指南（`selection.rst`、`plotting.rst`、`woe_iv.rst`）和 API 页。
- **领域稿 1.4 的功能 PR**（一个会话，`validation-tests`）：`_stats.py`、`calibration.py`、`metrics.py`、`vintage.py` 及其测试、API 页和指南，包括新指南 `validation.rst` 以及 `calibration.rst`、`metrics.rst`、`vintage.rst` 的小节。本稿的 PR 不改这些指南，画图 PR 只写 `plotting.rst`。领域功能如果要在 `plotting.py` 里加图，比如迁移矩阵热力图，等本稿的画图 PR 合并后再做。指南里的图在集成 PR 中加进 `docs/_plotting.py`。
- **只在集成 PR 里改：** README（中英文）、`docs/index.rst`、`user_guide/index.rst` 与 `api.rst` 的顺序、`compatibility.rst`、`glossary.rst`、`conventions.rst`、`migration.rst`、CLAUDE.md，以及 `tests/test_validation.py` 的登记表。两份稿子的功能 PR 都在描述里列出它们需要的改动。
- **共用文件：** `kivi.po` 是唯一大家都能改的文件，冲突用 `--merge-po` 解决。
- **合并与发布：** 由协调会话按你的指示，依依赖顺序合并；1.4.0 由你确认后发布。

## 文档与实施计划

1. 本稿审定：2026-11-23 已确认，下一节的 10 项全部采用推荐。
2. 两个功能会话并行：
    - **A：** `selection.py`，包括 `vif_filter` 和 `selection_path`，及测试、`selection.rst` 中的两个新小节和 API 页。
    - **B：** `plotting.py`，包括 `distribution_chart` 和 `precision_recall_chart`，及测试、`plotting.rst` 和 API 页。合并后，同一会话再开第二个 PR 做 `woe.py` 的 `woe_trend` 和 `woe_iv.rst` 的一个小节。
3. **集成 PR**，在本稿和领域稿的功能 PR 都合并后进行：
    - README 的 API 表；
    - `compatibility.rst`：`selection_path` 的列名，`woe_trend` 的标签；
    - 迁移指南更正：
        - `StepWise` 的真实语义，以及它与 `stepwise`、`selection_path` 的关系；
        - `LowVIFFeatures` 对应 `vif_filter`；
        - `plot_ks` 对应 `gains_chart`，`plot_score` 对应 `distribution_chart`，PR 曲线对应 `precision_recall_chart`；
        - `monotony` 对应 `woe_trend`；
        - 不做的四项，以及替代写法；
    - CLAUDE.md 和测试登记表；
    - 视需要在 `risk_workflow.rst` 或教程里用上 `vif_filter` 和 `selection_path`。
4. 发布 1.4.0，需要你的确认。master 上已有的三个文档片段一起发布。

## 决定

2026-11-23，以下 10 项全部采用推荐方案。

| # | 问题 | 决定 | 未采用的选项 |
| --- | --- | --- | --- |
| 1 | `vif_filter` 每轮剔除谁 | 剔除 VIF 最大的变量（教科书做法）。并列时，比如几个变量同为无穷大，剔除 `priority` 最低的 | B：旧版规则，剔除后其余 VIF 之和最小的那个；C：在超过阈值的变量中剔除 `priority` 最低的，即优先保 IV 高的变量，但不一定先降最大的 VIF |
| 2 | `threshold` 默认值 | 5，与旧版相同，也是两个常见经验值中较严的 | 10 |
| 3 | 旧 `StepWise` 的对应形式 | 新函数 `selection_path`，每步一行的报告表 | 给 `stepwise` 加 `criterion="auc"`，按 AUC 增量贪心选择（旧版并没有这种语义）；或不做 |
| 4 | 验证样本 | 支持可选的 `validation=(X_valid, target_valid)`，多出 `validation_*` 三列。开发样本上的 AUC 随变量增加几乎总是上升，验证样本才能看出增益在哪里停止 | 只在开发样本上计算，与旧版相同 |
| 5 | 提前停止 | 不内置，报告每一步，由使用者读表决定 | 加 `min_gain`，与旧版 `early_stop` 一样按相对增量停止 |
| 6 | 分布图的输入 | 任何有 `non_event`、`event` 列的表（`gains_table`、`woe_table`），与其他图“从 kivi 的表画”一致 | 直接接收分数和标签画直方图（旧版做法） |
| 7 | 精确率–召回率曲线图 | 做。很小，`roc_table` 已有数据；欺诈和极不平衡样本常用 | 不做 |
| 8 | WOE 趋势标签的形式 | 函数 `woe_trend(table)` | 另外给 `profile_table` 加 `woe_trend` 列，这样筛选时每个数值变量都有标签，但要改 `univariate.py`；或不做，因为 `monotonic_edges` 已能保证单调 |
| 9 | 相邻 WOE 相等 | 不打断趋势：相等的一段既不算升也不算降 | 严格单调，相等即 `"irregular"`；或像旧版一样算作上升 |
| 10 | 不做的四项（`int_score`、`top_n_features`、混淆矩阵图、等级 PD 图） | 只在迁移指南里写替代写法 | 挑其中几项实现 |
