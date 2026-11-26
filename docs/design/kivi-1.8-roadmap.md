# kivi 1.8 和 1.9 开发计划

1.7.0 已于 2026-11-25 发布（标签在 1701f2a5）。`kivi-1.4-risk-parameters.md` 规划的 1.5–1.7 已全部完成，之后没有排期。2026-11-25，用户选择把后续工作压缩成两个版本（“按两版改写”），各版内容如下：

- **1.8：** 全部新功能，由四个功能会话完成；
- **1.9：** 把最低 Python 版本提高到 3.11，并修复 1.8 发布后发现的问题。

价值偏低或需求未定的内容不排期，理由见第 5 节。

本稿依据四方面的材料：

1. 2026-11-24 对分箱和 WOE 算法完整性的分析，以及在银行数据上的三组实验（第 2 节）；
2. 1.4–1.7 设计稿中“以后再说”的事项；
3. 对 master 上各模块现状的核对；
4. 用户 2026-11-24 的新原则：“运行时只依赖 NumPy 和 pandas”不再是判断标准，依赖可以适当扩大。每个具体的新运行时依赖或 extra 仍由用户决定。本稿不新增任何依赖。

流程沿用 1.4–1.7：

- 开工前审定 API，需要依据监管或文献的部分先核对原文；
- 功能会话各自拥有互不重叠的文件，入口页由集成 PR 统一改写；
- 合并和发布都等用户的指示。

## 1. 版本划分

| 版本 | 会话 | 内容 |
| --- | --- | --- |
| **1.8** | `binning` | 第一个 PR：`optimal_edges`，`chimerge_edges` 和 `monotonic_edges` 的 `min_bin_share`，`fit_spec(method="optimal")`，以及指南里的样本外 IV。第二个 PR：连续目标的有监督分箱 |
| | `plotting-tables` | 1.4–1.7 新表格的五张图 |
| | `default-rates` | 一年违约率、长期平均违约率、对准中心趋势的校准 |
| | `rule-search` | 单变量候选规则的搜索 |
| | 集成 PR | 入口页、兼容性条目、术语表，以及预告 1.9 最低版本的 `changed` 片段 |
| **1.9** | 协调会话 | 最低 Python 版本提高到 3.11，同时修复 1.8 发布后发现的问题 |

**为什么是两个版本：**

- **最低版本要提前预告：** 按 `docs/development/releasing.rst`，提高最低版本的次版本，必须由前一个版本的 `changed` 片段预告。1.7 没有预告，所以最快 1.9 才能生效。
- **只有 `binning.py` 要排队：** 四个功能会话中，只有 `binning` 的两个 PR 改同一组文件，在同一会话内先后完成；其余会话的文件互不重叠，可以并行。
- **规模：** 1.8 有五个功能 PR，与 1.1 相当，是迄今最大的一次发布，所以不再往里加内容。

**2.0：没有计划。** master 上没有任何 `_deprecation.warn` 调用，也就没有待移除的弃用。

## 2. 依据：分箱实验

实验数据是银行数据（4521 行，521 个事件），代码不进仓库。

### 2.1 贪心合并已经接近最优

实验用动态规划求精确的 IV 最优分箱，分箱上限 5 箱，初始箱与 `chimerge_edges`、`monotonic_edges` 相同，都是 20 个分位箱。

- 不加约束时：动态规划最优与 `chimerge_edges(alpha=None)` 比较 6 次；
- 单调约束下：动态规划最优与 `monotonic_edges(max_bins=5)` 比较 6 次。

12 次比较中，11 次的 IV 完全相同，唯一的差异是 `balance`（0.1051 对 0.1061）。

**结论：** 精确最优算法带来的不是更高的 IV，而是能同时、有保证地满足多项约束。IV 可以按箱相加，前提是没有一个类别计数为 0 的箱，因此一维的情形用动态规划就能精确求解，不需要整数规划求解器。

### 2.2 现有方法缺约束

| 变量 | 方法 | 问题 |
| --- | --- | --- |
| `day` | `monotonic_edges` | 有一箱只占 1.3% |
| `campaign` | `monotonic_edges` | 有一箱只占 2.9% |
| `age` | `chimerge_edges` | 有一箱只占 3.8% |
| `duration` | `monotonic_edges(max_bins=None)` | 有一箱 0 个事件，WOE 完全来自 0.5 的平滑 |
| `duration` | `chimerge_edges` | 有一箱只有 1 个事件 |
| `age`、`balance`、`campaign` | `tree_edges`、`ks_edges` | 趋势为 `irregular` 或倒 U 形，不能要求单调 |
| `age` | `monotonic_edges` | 只能分 2 箱（IV 0.102）。不加单调约束的 ChiMerge 给出 3 箱、U 形（IV 0.126），`woe_trend` 能识别这种形状，但没有方法能按它拟合 |

`tree_edges` 和 `ks_edges` 有 `min_bin_share`，`chimerge_edges` 和 `monotonic_edges` 没有。所有有监督方法都不限制每箱的事件数。

### 2.3 有监督分箱后的 IV 偏高

实验对与目标无关的纯噪声变量重复了 200 次：

| 分箱 | 平均 IV | 95% 分位 | IV ≥ 0.02 的比例 |
| --- | --- | --- | --- |
| 等频 5 箱 | 0.0090 | 0.0226 | 6% |
| 等频 20 箱 | 0.0415 | 0.0646 | 97% |
| `chimerge_edges(alpha=None)`，5 箱 | 0.0251 | 0.0414 | 68% |
| `tree_edges`，5 箱 | 0.0409 | 0.0612 | 98% |

- **无监督分箱：** 均值符合近似式 (n_b·n_g/n)·IV ≈ χ²(k − 1)。`profile_table` 的快速分箱是 5 个分位箱，它的 `chi2_p_value` 因此是成立的。
- **有监督分箱：** 边界是看着目标选出来的，5 箱的偏差相当于 20 个自由度。纯噪声几乎总能过 0.02 这个常用的“无预测力”门槛。
- **处理方式：** 用验证样本的 IV 或置换检验来筛选变量。
- **不改编码：** 折外 WOE 不适合作为最终编码，因为它让训练时的 WOE 与评分卡的分数表对不上，破坏了“每箱一个分数”。

## 3. 1.8

### 3.1 会话 `binning`

**拥有的文件：**

- `binning.py`、`spec.py`；
- `tests/test_supervised_binning.py`、`tests/test_spec.py`；
- `binning.rst`、`woe_iv.rst`、`docs/api/binning.rst`；
- 发布说明片段。

两个 PR 先后进行。第二个 PR 在第一个合并后，从最新的 master 分出。

#### 第一个 PR：约束分箱

```python
# kivi.binning
optimal_edges(values, target, *, max_bins=5, min_bin_share=0.05, min_bin_events=1,
              trend="auto", initial_bins=20, special=None, weights=None)
# -> edges，与其他 *_edges 相同：两端为 -inf、inf

chimerge_edges(..., min_bin_share=0.0)    # 新增关键字，默认值保持现有结果
monotonic_edges(..., min_bin_share=0.0)   # 同上

# kivi.spec
fit_spec(..., method="optimal")           # 新的 method 取值
```

**`optimal_edges` 的设计：**

- **目标：** 在初始箱上取 IV 最大的划分。初始箱的规则与 `chimerge_edges` 相同：`initial_bins` 个分位箱；不同取值不多于这个数时，每个取值一箱。
- **约束：**
  - 最多 `max_bins` 箱；
  - 每箱至少占拟合值（或权重）的 `min_bin_share`；
  - 每箱的事件数和非事件数都至少为 `min_bin_events`，单位是权重；
  - WOE 满足 `trend`。
- **`min_bin_events` 必须为正，默认 1：** 每箱两类都有，IV 可以按箱相加，结果与 `woe_table` 算出的 IV 一致，也排除了 2.2 中 0 个事件的箱。因为不会出现 0 计数的格子，所以没有 `smoothing` 参数，`tests/test_validation.py` 的 smoothing 清单不用改。
- **`trend`：**
  - 取值沿用 `woe_trend` 的标签：`"increasing"`、`"decreasing"`、`"u_shaped"`、`"inverted_u"`，另加 `"any"`（不约束形状）和 `"auto"`。参数不叫 `monotonic_edges` 的 `direction`，因为 U 形不是一个方向。
  - `"auto"` 与 `monotonic_edges` 相同：在 increasing 和 decreasing 中取 IV 较大的一个，相等时取 increasing。
  - U 形和倒 U 形只在显式指定时才用。原因是这两种形状的 IV 总不低于单调，如果由 `"auto"` 选，几乎不会选中单调；而 U 形是业务判断，比如 `age`。
- **算法：** 动态规划。状态是“最后一箱 + 已用箱数”，单调和单峰的约束通过相邻箱的 WOE 比较来检查。计算量为 O(n²·k) 次长度为 n 的向量运算，初始箱数到 200 都很快。
- **并列：** 相对差在 1e-10 以内的 IV 视为相等，取在前的划分，与 `_first_max` 相同。
- **可行性：** 只有一箱时总满足占比和形状约束，所以只要事件数和非事件数都不少于 `min_bin_events`，问题就有解；否则报错。
- **参考：**
  - 小规模时与穷举所有划分比较，结果须完全一致。这是精确性的证明。
  - 在临时环境里用 optbinning 1.1 算一次参考值，传入相同的 `user_splits`，即 kivi 的初始箱，然后把结果写进测试，与 1.7 处理 R 的 `EMP` 包相同。optbinning 不进 `test` extra，原因有二：它要求 Python ≥ 3.11；它依赖的 ortools 要求 NumPy ≥ 2.0.2，1.9 把最低版本任务换到 Python 3.11 后，该任务仍要安装 NumPy 1.24，依赖装不上。

**`chimerge_edges` 和 `monotonic_edges` 的 `min_bin_share`：**

- 合并开始前，先把占比最小的箱并入卡方统计量较小的那个邻箱，规则与 `chimerge_groups` 的 `min_group_share` 相同。
- 默认值 0 保持现有结果不变。

**样本外 IV：** 只写进指南，不新增 API。

- `woe_iv.rst` 增加“开发样本 IV 偏高”一节，写明 2.3 的现象、无监督分箱下的 χ² 近似，以及两种写法：
  - 验证样本 IV：`woe_table(spec.cut(valid)[column], valid_target)`；
  - 置换检验：打乱目标，重拟合分箱，比较 IV。
- `binning.rst` 的 Limits 一节加一句提醒。
- **为什么不加 API：** 验证样本 IV 用现有函数一行就能算；置换检验要按方法重新拟合分箱，会重复 `fit_spec` 内部的逻辑。以后再加 API 不破坏兼容，加了再删就会。

#### 第二个 PR：连续目标的有监督分箱

1.6 的 `lgd_table` 仍用 `bin_edges` 分组；当时“按方差下降的有监督分箱”留待以后。最小化箱内平方和同样可以按箱相加，所以复用第一个 PR 的动态规划。

```python
# kivi.binning
variance_edges(values, response, *, max_bins=5, min_bin_share=0.05, trend="auto",
               initial_bins=20, special=None, weights=None)   # 暂名
# -> edges，交给 cut，再交给 lgd.lgd_table
```

- **`response`：** 有限实数，如已实现的 LGD 或 CCF，可以超出 [0, 1]，与 `kivi.lgd` 一致。参数不叫 `target`，因为 kivi 的 `target` 总是 0 与 1。
- **目标：** 箱内加权平方和最小，即箱间方差最大。
- **`trend`：** 约束各箱的均值，取值和 `"auto"` 的规则同 `optimal_edges`。
- **参考：**
  - 与穷举所有划分比较；
  - 初始箱取每个不同值一箱、只分两箱时，划分与 scikit-learn `DecisionTreeRegressor(max_depth=1, min_samples_leaf=...)` 的切分相同。比较的是划分，而不是阈值，因为 scikit-learn 的阈值取两个值的中点。

### 3.2 会话 `plotting-tables`

**拥有的文件：** `plotting.py`、`tests/test_plotting.py`、`plotting.rst`、`docs/api/plotting.rst`，以及发布说明片段。

1.1 的路线图写的是“每张结果表都有图”。1.4–1.7 新增的表格还没有图，五张全做：

| 函数（暂名） | 画什么 | 表格 |
| --- | --- | --- |
| `survival_chart` | 按期的 `cumulative_pd` 或 `survival` 阶梯线，竞争风险时画出两类 | `lifetime.survival_table` |
| `transition_chart` | 迁移矩阵热力图，对角线高亮 | `vintage.transition_matrix`、`roll_rates` |
| `rule_set_chart` | 规则瀑布：每条规则的 `new_hits` 柱，叠加累计 `approval_rate` | `rules.rule_set_table` |
| `profit_chart` | `profit` 对 `approval_rate` 的曲线，标出利润最大的截断点 | `strategy.cutoff_table`（含 `revenue`） |
| `lgd_chart` | 各组的平均 LGD，加 `zero_share` 和 `full_share` | `lgd.lgd_table` |

约定同现有的图：

- 表格作为第一个参数，关键字参数 `ax`，返回 Axes；
- 只检查列；
- 不调用 `show`；
- 图中文字只用列名和值。

### 3.3 会话 `default-rates`

**拥有的文件：**

- `vintage.py`、`calibration.py`；
- `tests/test_vintage.py`、`tests/test_calibration.py`、`tests/test_typing.py`；
- `vintage.rst`、`calibration.rst`、`docs/api/vintage.rst`、`docs/api/calibration.rst`；
- 发布说明片段。

`tests/test_typing.py` 归这个会话，因为四个会话中只有它的函数按输入决定返回类型，需要重载声明。

原文核对、修订后的 API、测试依据和待定事项见 `docs/design/kivi-1.8-default-rates.md`。

### 3.4 会话 `rule-search`

**拥有的文件：** `rules.py`、`tests/test_rules.py`、`rules.rst`、`docs/api/rules.rst`，以及发布说明片段。

```python
# kivi.rules
rule_candidates(frame, target, *, columns=None, method="tree", max_bins=5,
                min_hit_rate=0.01, min_lift=2.0, special=None, weights=None)   # 暂名
# -> DataFrame，索引 rule：variable、operator、threshold、hits、hit_rate、event、event_rate、lift
```

- **候选阈值：** 每个数值列用公共函数 `binning.tree_edges` 或 `ks_edges` 拟合切点，`rules.py` 不引用 `binning.py` 的私有函数。
- **规则：** 每个切点取事件率较高的一侧作为一条规则（`>` 或 `<=`）。保留命中率不低于 `min_hit_rate`、提升度不低于 `min_lift` 的规则，按 `lift` 从高到低排序。
- **特殊值：** 不命中阈值规则；特殊值需要的话，可以单独写成一条规则。
- **衔接：** 选中的规则写成 `frame[variable] > threshold` 这样的布尔列，直接交给 `rule_table` 和 `rule_set_table`。
- **局限：** 这些是开发样本上的候选，和 2.3 的 IV 一样会偏乐观。指南要求在验证样本上用 `rule_table` 复核。

### 3.5 集成与发布

- **集成 PR：**
  - README 表格和路线图（中英两版）、首页和指南卡片；
  - `compatibility.rst` 的新函数和新列、术语表、CLAUDE.md；
  - 风险流程指南里的约束分箱，以及各功能 PR 描述的示意图；
  - 新的交叉核对移入 `test_reference.py`。
- **预告最低版本：** 加一个 `changed` 片段，写明 1.9 起要求 Python 3.11。
- **发布 1.8.0：** 等用户的指示。

## 4. 1.9

由协调会话完成，内容很少：

- **最低版本：**
  - `pyproject.toml` 改为 `requires-python = ">=3.11"`（分类器没有列出具体的 Python 版本，不用改）；
  - `.github/workflows/docs.yml` 的 `minimum-versions` 任务改用 Python 3.11，仍然是 NumPy 1.24.4 和 pandas 2.0.3，两者都有 cp311 的 wheel；
  - `getting_started/install.rst` 和 `api/compatibility.rst` 的版本一起改，按 `releasing.rst` 的 Minimum versions 一节。
- **理由：** Python 3.10 已于 2026 年 10 月停止支持。NumPy 1.24 和 pandas 2.0 不动：提高它们的收益很小，银行的环境又往往偏旧，等有具体需要时再按同样的规则预告。
- **修复：** 1.8 发布后发现的问题。
- **发布 1.9.0：** 等用户的指示。

## 5. 不排期的内容

| 内容 | 不排期的理由 |
| --- | --- |
| 二维分箱 | 工作量大：要用整数规划（如 SciPy 的 `milp`），`BinningSpec` 要升到格式版本 3，`woe_transform` 要支持列对。需求证据也弱：评分卡里的交互通常靠分群，或者先造交叉特征（如负债收入比）再按一维分箱 |
| Cox 比例风险模型 | `lifetime.rst` 已有离散时间风险模型的写法；需要 Cox 的用户可以直接用 statsmodels 的 `PHReg` |
| beta 回归 | 只能处理严格在 (0, 1) 内的值，而 LGD 在 0 和 1 处有点质量，1.6 的 fractional logit 正好能处理 |
| 下行 LGD | 主要是机构自己的判断，`lgd.rst` 已写明由银行自行处理 |
| 多期 Pluto–Tasche | 很少用，又要做蒙特卡洛 |
| 多变量规则树 | 需要一棵新的多变量树，规模大；先看单变量搜索的使用情况 |
| 多期正态检验、带相关的二项检验、红绿灯法 | 1.4 决定等实际需求再做。kivi 按 ECB 的验证报告来做，ECB 用的是逐等级的 Jeffreys 检验，这已经有了；红绿灯法主要是德国央行的做法 |

这些内容以后有了实际需求，再按同样的流程立项。

## 6. 维持不做

| 内容 | 理由 |
| --- | --- |
| 市场风险、组合信用模型和经济资本、定价和 RAROC 函数 | 用户群和口径不同（2026-11-23 的决定），与依赖无关 |
| 机器学习模型（GBDT、孤立森林）、图网络反欺诈和 AML | “不做模型包装”的决定本身仍然成立。kivi 的指标、校准和策略函数接受任何分数，scikit-learn 或 LightGBM 的模型不用包装就能接进来 |
| K-means / Jenks 分箱 | 2026-11-22 定为推迟，如果做，用确定性的一维算法（Fisher 的动态规划）。它只按数值聚类，不解决新问题 |
| CAIM、Ameva、Khiops | 信用风险实务很少用，没有证据比现有方法好 |
| 多分类 WOE | 评分卡惯例是二分类；多状态已有 `transition_matrix` |
| 折外 WOE 作为最终编码 | 破坏“每箱一个分数”（2.3） |
| Polars 等其他数据框输入 | 牵涉每个函数的输入检查；转换只需一行 `to_pandas()` |
| 输出下限（72.5%） | 需要标准法的 RWA（1.5 决定 7） |

## 7. 决定

| # | 问题 | 状态 | 内容 |
| --- | --- | --- | --- |
| 1 | 版本划分 | 2026-11-25 已定（“按两版改写”） | 两个版本：1.8 放全部功能，1.9 提高最低版本；第 5 节的内容不排期 |
| 2 | `optimal_edges` 的 API | 2026-11-25 已定（“都按推荐”） | 第 3.1 节：参数名 `trend`，取值沿用 `woe_trend` 的标签；`"auto"` 只在两个单调方向中选；`min_bin_events` 必须为正，默认 1 |
| 3 | optbinning | 2026-11-25 已定（“都按推荐”） | 在临时环境里算一次参考值，写进测试，不进 `test` extra |
| 4 | 样本外 IV | 2026-11-25 已定（“都按推荐”） | 只写指南 |
| 5 | 图 | 2026-11-25 已定（“都按推荐”） | 第 3.2 节的五张全做 |
| 6 | 最低版本 | 2026-11-25 已定（“都按推荐”） | 只提高 Python，到 3.11；1.8 预告，1.9 生效 |
| 7 | 多期检验和红绿灯法 | 2026-11-25 已定（“都按推荐”） | 继续等实际需求 |
| 8 | 二维分箱 | 2026-11-25 已定（“都按推荐”） | 不排期，所以 SciPy 是否进入运行时依赖的问题暂不需要决定 |
| 9 | 机器学习模型 | 2026-11-25 已定（“都按推荐”） | 维持不做 |
| 10 | 新函数的名称 | 2026-11-25 已定 | 全部采用：`optimal_edges`、`variance_edges`（`kivi.binning`），`rule_candidates`（`kivi.rules`），`survival_chart`、`transition_chart`、`rule_set_chart`、`profit_chart`、`lgd_chart`（`kivi.plotting`），`default_rate_table`（`kivi.vintage`），`central_tendency_shift`、`quasi_moment_fit`（`kivi.calibration`；见 `kivi-1.8-default-rates.md` 决定 1）。第 3 节代码中标注的“暂名”是审定前的写法 |

## 8. 协调

- **本稿：** 只新增 `docs/design/kivi-1.8-roadmap.md`，与其他会话不冲突。
- **进行中的会话：** “Kivi 风控包 skills 设计”会话正在运行，它的分支还没有推送；本稿不对它的内容做任何假设。
- **1.8 的文件归属：** 四个会话的文件见第 3.1–3.4 节，互不重叠。`pyproject.toml`、`test_reference.py`、`_plotting.py`、`_diagrams.py` 和各入口页都留给集成 PR，`kivi.po` 照例由各 PR 共同修改，冲突用 `--merge-po` 解决。
- **开工顺序：**
  1. 本稿审定后，`binning`、`plotting-tables` 和 `rule-search` 可以立即开工；
  2. `default-rates` 先核对原文，再审定 API；
  3. `binning` 的第二个 PR 在第一个合并后开始。
