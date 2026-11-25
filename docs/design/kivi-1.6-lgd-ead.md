# kivi 1.6 实施计划：LGD 与 EAD

2026-11-24 起草 · 待审定：第 6 节的 8 项

## 1. 目标与范围

1.6 的内容已由 `docs/design/kivi-1.4-risk-parameters.md` 第 6 节设计，2026-11-23 审定。该稿第 10 节第 12 项决定清收 LGD 不截断、日计数用 ACT/365，第 13 项决定 `fractional_regression` 用稳健（HC0）标准误。内容如下：

- **新模块 `kivi.lgd`：**
  - `workout_lgd`：清收 LGD；
  - `conversion_factor`：已实现的 CCF；
  - `lgd_table`：连续目标的分组表；
  - `fractional_regression`：fractional logit。
- **`kivi.datasets`：** 模拟违约账户及其清收现金流。
- **验证：** 1.4 的 `backtest_table` 和 `generalized_auc` 直接用于 LGD 和 CCF。

与 1.5 一样，开工前先把草稿与监管原文逐条核对，再审定 API。本稿做三件事：

1. 记下核对结果和出处（第 2 节）；
2. 据此修订 API（第 3 节）；
3. 给出测试依据、会话分工和待你决定的事项（第 4–6 节）。

1.6 不需要新的运行时依赖：`fractional_regression` 沿用 `_logit` 的迭代加权最小二乘，现金流的折现和分组表只用 NumPy 和 pandas。

## 2. 原文核对结果

2026-11-24 核对，依据的文本如下。

- **EBA/GL/2017/16**（PD 估计、LGD 估计和违约敞口处理的指引）第 6.3 节。
- **巴塞尔框架**，2023 年 1 月 1 日起生效的版本：
  - CRE32（下限、初级法 CCF、可自估 CCF 的范围）；
  - CRE36（LGD 和 EAD 自估的要求）；
  - CRE20（标准法 CCF）。
- **CRR3**（Regulation (EU) 2024/1623）第 4(1)(56)、166、182 条。EUR-Lex 拒绝自动访问，取自 Wayback 快照。
- **EBA/CP/2025/10**（2025 年 7 月 2 日），CCF 估计指引的征求意见稿。截至核对时，EBA 页面上只有未附链接的 "Final Report" 条目，未见正式版。CRR3 第 182(5) 条要求在 2026 年底前发布。
- **ECB**，"Instructions for reporting the validation results of internal models – IRB Pillar I models for credit risk"（2019 年 2 月）第 2.3、2.6、2.9 节。未见更新版本；2024 年的门户填报说明只讲操作。
- **中国：** 《商业银行资本管理办法》（国家金融监督管理总局令 2023 年第 4 号）第九十二、九十三条，附件 5 第六部分（四）（LGD）和（五）（EAD）。gov.cn 无法访问，取自 Wayback 快照。

### 2.1 LGD

| 项目 | EBA / 巴塞尔 | 中国 2023 | 对 API 的影响 |
| --- | --- | --- | --- |
| 定义 | 已实现 LGD 是经济损失与违约时未偿金额之比（EBA 131）。经济损失是未偿金额加上重大的直接和间接清收费用，减去折现到违约时点的回收（EBA 132）。来源不明的回收也计入（EBA 133）。巴塞尔：必须用经济损失而不是会计损失，包括重大的折现效应和直接、间接费用（CRE36.76） | 损失金额占违约债项风险暴露的比例（附件 5 六(四)1），风险暴露含本金和应计未付的利息、费用（6(3)）；基于经济损失，含直接和间接成本（2） | 与草稿的公式一致 |
| 折现率 | 年利率：违约时点适用的 3 个月 EURIBOR（或该币种可比的流动性利率）加 5 个百分点，回收、费用和追加提款都折到违约时点（EBA 143）。利率随每笔敞口的违约日和币种而不同 | 把回收折现到违约时点，折现率反映清收期间持有违约债项的成本；回收不确定时加风险溢价，确定时可用无风险利率（2(3)） | **`rate` 要能逐笔给出**，草稿只有一个数 |
| 日计数 | 未规定，只说"年利率" | 未规定 | ACT/365（已审定第 12 项）作为约定写进文档 |
| 违约前的费用 | 违约前发生的重大清收费用也要计入，除非已在未偿金额中或属于更早的违约（EBA 144–146） | 未见专门规定 | **草稿"违约日之前的现金流报错"与此冲突**（决定 3） |
| 追加提款 | 违约后的追加提款一律计入分子（EBA 142）；若 CCF 已包含追加提款，分母也加上它们（EBA 140），零售未作此处理的不加（EBA 141） | 未见专门规定 | 带符号的一笔现金流无法同时进分子和分母（决定 4） |
| 违约后资本化的利息和费用 | 不增加经济损失，但其回收照常计入（EBA 137–138） | 未见专门规定 | 调用者不传入这些金额；文档说明 |
| 核销 | 违约时或之前核销的金额加到违约时的未偿金额，分子分母都加（EBA 134） | 未见专门规定 | 调用者把它加到 `ead`；文档说明 |
| 治愈 | 视同借款人在治愈日偿还了当时的未偿金额，加一笔"人为回收"，同样折现；多次违约从第一次违约起算（EBA 135） | 附件 5 未见 | 调用者加一笔现金流；文档和指南示范 |
| 未结案 | 观察平均只用已结案的清收，不含预期的未来回收（EBA 154），结案的定义见 EBA 157；长期平均再对未结案做调整（EBA 158）；不得剔除观察期内的任何违约（EBA 163）。ECB 回测只用已结案的（2.3(i)、2.6.1(d)） | 银行须设定并记录"清收完成时间点"（六(四)7） | 没有现金流的账户 LGD 为 1，只对已结案的账户成立（决定 5） |
| 长期平均 | 按违约数的算术平均（EBA 150），不得按年平均，零售可给近期数据更高权重（EBA 150–152）。巴塞尔：长期违约加权平均（CRE36.83、36.81）。ECB 报告按数目加权的平均 | 不低于长期违约加权平均损失率（六(四)4） | `lgd_table` 的不加权均值正确；`weights` 只用于有依据的加权，不用 EAD 加权 |
| 负值与超过 100% | 有盈利时，计算观察平均和长期平均的已实现 LGD 逐笔记为 0，不相互抵消；未截断的值可用于风险区分（EBA 160）。远高于 100% 的值保留，应找原因或单独分段（EBA 162） | 小于 0 的样本按 0 处理（六(四)15） | `workout_lgd` 不截断正确（已审定第 12 项）；**`lgd_table` 计算平均前应把负值记为 0**（决定 6） |
| LGD 下限 | 下限用于参数估计值，不用于已实现 LGD：公司无担保 25%，住房抵押 5%，合格循环零售 50%，其他零售无担保 30%（CRE32.16、32.58–59） | 同，但个人住房抵押为 10%（第九十二条） | 不进入 1.6 的函数；指南列出 |

### 2.2 EAD 和 CCF

| 项目 | 巴塞尔 / 欧盟 | 中国 2023 | 对 API 的影响 |
| --- | --- | --- | --- |
| 可自估 CCF 的范围 | 只限未提用的循环承诺，且标准法 CCF 不是 100%（CRE32.36、零售 CRE32.63；CRR3 166(8b)） | 可循环类表外项目可自估，非循环类用第八十二条的 CCF，100% 的仍为 100%（第九十三条三、四） | 指南说明 |
| EAD 下限 | 表内金额加表外金额按标准法 CCF 的 50%（CRE32.36，零售 CRE32.64；CRR3 166(8c)，中央政府除外） | 公司：取内部估计与上述两项之和的较大值（第九十三条三）；零售未见下限（第九十三条四），与巴塞尔不同 | 由调用者施加；指南列出 |
| 参考日 | 12 个月固定期限法，借款人和债项特征取违约前 12 个月（CRE36.93；CRR3 182(1)(g)；EBA CP §59）。放款后 12 个月内违约的，取未提用额度严格为正的最早日期（EBA CP §60） | 应使用 12 个月固定期限法，能证明审慎的可用其他方法（附件 5 六(五)12） | 选参考日是调用者的数据步骤；文档说明 |
| 已实现 CCF | 未提用额度因子：(违约时提用 − 参考日提用) / (参考日额度 − 参考日提用)（CRE36.95 脚注 7；EBA CP §61(a)）。ECB 按银行模型的定义计算（2.9.1(b)） | 同类口径（违约加权长期平均，六(五)3） | 与草稿的公式一致 |
| 额度 | 已告知额度与未告知额度中较大者（CRR 4(1)(56)）；参考日之后的额度提高不进分母，可以得到 200% 的 CCF（EBA CP 例 3） | 未见专门规定 | 文档说明 `limit` 取参考日的值 |
| 满额或超额使用 | 使用率达到或超过 100% 时改用 违约时提用 / 参考日额度（EBA CP §61(b)），银行也可设定使用率阈值（§61(c)、§73）。接近满额时因子不稳定（CRE36.95；中国六(五)14）。ECB 对按 EAD 法处理的债项不做 CCF 检验，改做 EAD 回测（2.9.1(g)、2.9.3） | 关注接近满额时未提用额度因子的不稳定 | **草稿只在 limit == drawn 时给 NaN，drawn > limit 时分母为负，结果的符号会悄悄反转**（第 3.2 节修正；决定 7） |
| 负值 | 量化时负的已实现 CCF 记为 0，未截断的值可用于风险区分（CRR3 182(1)；EBA CP §99）。ECB 报告被截断的比例 | 未见专门规定 | 不截断；文档示范 `np.maximum(ccf, 0)` |
| 上限 | 不得截断在 100%，参考数据不得封顶（CRE36.95–96；CRR3 182(1d)；EBA CP 背景 §22） | 不应设置本金上限（六(五)15） | 不截断，与草稿一致 |
| 初级法 CCF | 与标准法相同：100%、50%、40%（承诺）、20%、10%（可无条件撤销）（CRE32.33、CRE20.95–100）；75% 是旧值 | 第九十三条二引第八十二条：100%，贷款承诺 40%（可无条件撤销 10%），未使用的信用卡额度 40% 或 20%，NIF/RUF 50% | 指南列出 |

## 3. API：在 1.4 稿第 6 节基础上的修订

```python
# kivi.lgd（新模块）
workout_lgd(loan, default_date, ead, flow_loan, flow_date, flow_amount, *, rate)
conversion_factor(ead, drawn, limit)
lgd_table(groups, lgd, *, floor=0.0, weights=None)
fractional_regression(X, response, *, weights=None)

# kivi.datasets
make_defaults(*, n_defaults=2000, rng=None)
make_recoveries(defaults, *, rng=None)
```

### 3.1 `workout_lgd`

- **参数名（决定 1）：** 改用 `loan` 和 `flow_loan`，与 `delinquency_panel(loan, …, payment_loan, …)` 和 `vintage_table(loan, …)` 一致。参数顺序不变：先是每笔贷款一行的 `loan`、`default_date`、`ead`，再是每笔现金流一行的 `flow_loan`、`flow_date`、`flow_amount`。日期的读法与 `delinquency_panel` 相同。
- **`rate`：** 一个数，或者每笔贷款一个（与 `loan` 按位置或索引对应），对应 EBA 143 按违约日确定的利率。年利率，ACT/365：

  ```
  PV_j = flow_j · (1 + rate)^(−(t_j − t_D)/365)
  ```

- **结果（决定 2）：** 改为每笔贷款一行的 DataFrame，索引是 `loan` 的取值（索引名 `loan`），顺序同输入。列为：
  - `ead`；
  - `recovery`：正现金流的现值；
  - `cost`：负现金流的现值，记为正数；
  - `lgd`：(ead + cost − recovery) / ead。

  草稿只返回名为 `lgd` 的 Series。分开的列便于核对费用和回收，也是 EBA 144–146 和中国附件 24 要求验证的内容。
- **违约前的现金流（决定 3）：** 负的现金流（费用）可以早于违约日，按同一利率复利到违约日（指数为正），对应 EBA 144。正的现金流早于违约日仍然报错，因为违约前的还款不属于清收。
- **追加提款（决定 4）：** 作为负现金流计入分子，与费用相同。EBA 140 要求 CCF 包含追加提款时把它们的现值也加到分母，这由调用者把现值加到 `ead` 来完成，文档给出写法。1.6 不加区分现金流种类的参数。
- **未结案（决定 5）：** 函数不知道清收是否结束，对每笔贷款都给出到目前为止的结果。没有现金流的贷款 `recovery` 和 `cost` 为 0，`lgd` 为 1。docstring 和指南说明：
  - 已实现 LGD 只对已结案的贷款成立（EBA 154、ECB 2.6.1）；
  - 结案日由银行按 EBA 157 或中国附件 5 六(四)7 设定；
  - 回测前先筛选已结案的贷款。
- **文档说明的调用者职责：**
  - 违约后资本化的利息和费用不传入（EBA 137–138）；
  - 违约时或之前核销的金额加到 `ead`（EBA 134）；
  - 治愈时加一笔等于治愈日未偿金额的现金流（EBA 135）。
- **报错：**
  - 重复的 `loan`；
  - `flow_loan` 中有 `loan` 以外的贷款；
  - `ead` 不为正；
  - 缺失值；
  - 布尔值的金额；
  - 违约前的正现金流。

### 3.2 `conversion_factor`

- 公式不变：(ead − drawn) / (limit − drawn)，`drawn` 和 `limit` 取参考日的值，与 `like_input` 形状一致，Series 名为 `ccf`。
- **修正：** `limit <= drawn`（满额或超额使用）时结果为 NaN，而不只是 `limit == drawn`。超额时分母为负，草稿的公式会给出符号相反的 CCF。
- **满额使用（决定 7）：** 1.6 不实现 EBA CP §61(b) 的额度因子，因为该稿尚未定稿。文档给出 `ead / limit` 的写法，并说明 ECB 把这类债项改做 EAD 回测。
- **负值和超过 1 的值原样返回。** 文档示范量化时用 `np.maximum(ccf, 0)`（CRR3 182(1)），并说明参考日（12 个月固定期限）和额度的取法。缺失值返回 NaN，布尔值报错。

### 3.3 `lgd_table`

- **下限（决定 6）：** 新增 `floor=0.0`。计算各列之前，先把低于下限的 LGD 记为下限，对应 EBA 160 和中国六(四)15；`floor=None` 保留原值，用于风险区分。
- **列：**
  - `total`、`total_share`；
  - `mean`、`std`、`min`、`max`：都按施加下限后的值计算；
  - `zero_share`：LGD ≤ 0 的份额；
  - `full_share`：LGD ≥ 1 的份额。
- **其他规则：**
  - 均值不加权即按违约数平均，与 EBA 150、CRE36.83、中国六(四)4 一致；
  - `weights` 是频数权重，文档说明不要用 EAD 加权；
  - 分组规则同 `woe_table`（`_counts.group_counts`），索引 `bin`。

### 3.4 `fractional_regression`

- **结果：** 与 `logistic_regression` 相同的系数表，索引 `variable`，列 `coef`、`se`、`z`、`p_value`，`attrs` 含拟似然值和 `converged`。标准误为 HC0 夹心估计（已审定第 13 项）。
- **权重：** 频数权重下，"肉"是 Σ f_i s_i s_iᵀ。已核对 statsmodels 0.15 的 `GLM(Binomial, freq_weights=…).fit(cov_type="HC0")` 与重复行完全一致，可作参考实现。
- **`response`：** 必须在 [0, 1] 内，可以恰为 0 或 1。
- **`_logit.fit`：** 改为接受分数目标，梯度为 Xᵀw(y − p)。分离检测改为只看恰为 0 或 1 的观测：当它们被完全分开、而 (0, 1) 内没有观测阻止系数发散时，报 `SeparationError`。`logistic_regression` 等现有调用的结果必须逐位不变，这由已有测试保证。

### 3.5 模拟数据（决定 8）

仿照 `make_applications` 和 `make_transactions`，用两个函数，每个返回一张表。

- **`make_defaults(*, n_defaults=2000, rng=None)`：** 每笔违约贷款一行，列包括：
  - `loan_id`、`segment`（如有押品的贷款、无押品的贷款、循环额度）、`default_date`、`ead`；
  - 循环额度在参考日的 `limit` 和 `drawn`（其他贷款为缺失值）；
  - 押品价值；
  - 清收结束日 `closing_date`（数据截止时未结案的为缺失值）。
- **`make_recoveries(defaults, *, rng=None)`：**
  - 列：`loan_id`、`date`、`amount`、`kind`（`recovery`、`cost`、`cure`）。
  - LGD 呈双峰：有押品的大多接近 0，部分治愈；无押品的接近 1。
  - 含少量违约前的法律费用、回收超过 EAD 的负 LGD 和费用较高时超过 1 的 LGD，供文档示范第 2.1 节的规则。
  - 循环额度的违约时提用由同一模型生成，供 `conversion_factor` 使用。

## 4. 测试与参考

| 内容 | 对照 |
| --- | --- |
| `workout_lgd` | 手算小例子（单利率和逐笔利率、违约当天的现金流、违约前费用的复利、ACT/365）；与逐笔循环的暴力实现在随机数据上比对；报错和形状 |
| `conversion_factor` | 闭式值；`limit <= drawn` 为 NaN；Series 和数组的形状 |
| `lgd_table` | 与施加下限后的 pandas `groupby` 比对；整数权重等于重复行；`floor=None`；`zero_share` 和 `full_share` |
| `fractional_regression` | statsmodels `GLM(Binomial, freq_weights=…).fit(cov_type="HC0")` 到 1e-10；二元响应时系数等于 `logistic_regression`；整数权重等于重复行；`SeparationError` 的边界情形 |
| `_logit` 的修改 | `test_selection.py` 和 `tests/test_reference.py` 原样通过 |
| 模拟数据 | 同一 `rng` 可复现；`workout_lgd` 对模拟现金流的结果与模拟时的设定一致；结案和治愈的比例；双峰 |

功能 PR 的交叉验证先放在各自的测试文件里，用 `pytest.importorskip`，集成 PR 再移入 `tests/test_reference.py`。

## 5. 会话与流程

| 会话 | PR | 文件 |
| --- | --- | --- |
| `recoveries` | 一个小 PR，先合并 | `src/kivi/datasets/_recoveries.py`（新）、`src/kivi/datasets/__init__.py`、`tests/test_datasets.py`、`docs/user_guide/datasets.rst` 的新小节、`docs/api/datasets.rst`、发布说明片段 |
| `lgd` | 一个 PR，与 `recoveries` 并行开工 | `src/kivi/lgd.py`（新）、`src/kivi/_logit.py`、`tests/test_lgd.py`（新）、`docs/user_guide/lgd.rst`（新）、`docs/api/lgd.rst`（新）、发布说明片段；若 `_logit` 的修改需要，可改 `tests/test_selection.py` |

两个会话的文件互不重叠，`kivi.po` 用 `--merge-po` 解决冲突。`lgd` 会话先写代码和测试；指南的示例用模拟数据，等 `recoveries` 合并后再写。

流程：

1. 本稿审定。
2. 两个会话开工。我逐个评审，按你的指示合并。
3. 集成 PR 由我来写：
   - README、首页和指南索引：`lgd` 指南的位置由集成 PR 定，建议放在 `validation` 之前或 `capital` 之前；
   - `compatibility.rst`：`workout_lgd` 和 `lgd_table` 的列，`ccf` 的 Series 名，两张模拟表的列；
   - glossary 与 conventions：
     - 新参数 `flow_loan`、`flow_date`、`flow_amount`、`response`、`floor`；
     - 术语：经济损失、清收、结案、治愈、长期平均 LGD、已实现 CCF、参考日、12 个月固定期限、未提用额度因子、fractional logit、夹心估计；
   - 测试登记表，以及交叉验证移入 `test_reference.py`；
   - 图：LGD 的双峰分布、各分段的 LGD 表；
   - CLAUDE.md；
   - 指南中用 `backtest_table` 和 `generalized_auc` 回测 LGD 和 CCF 的示例，若功能 PR 未写。
4. 发布 1.6.0，需要你的确认。

1.7（`kivi.rules`、收益、EMP）在 1.6 发布后按同样的流程开工。

## 6. 决定

以下 8 项待审定，推荐方案在前。

| # | 问题 | 推荐 | 其他选项 |
| --- | --- | --- | --- |
| 1 | `workout_lgd` 的参数名 | `loan`、`flow_loan`，与 `delinquency_panel`、`vintage_table` 一致 | 草稿的 `account`、`flow_account` |
| 2 | `workout_lgd` 的结果 | 每笔贷款一行的 DataFrame：`ead`、`recovery`、`cost`、`lgd`，索引 `loan` | 草稿的 Series `lgd` |
| 3 | 违约前的现金流 | 费用（负值）复利到违约日，回收（正值）报错（EBA 144） | 草稿：一律报错 |
| 4 | 违约后的追加提款 | 作为负现金流计入分子；EBA 140 的分母调整由调用者加到 `ead`，文档给出写法 | 增加现金流种类参数，函数内处理分母 |
| 5 | 未结案的清收 | 照常计算，文档和指南要求回测前筛选已结案的（EBA 154、ECB 2.6.1） | 增加 `closed` 参数，未结案的返回 NaN |
| 6 | `lgd_table` 的负值 | `floor=0.0`，计算各列前把负值记为 0（EBA 160、中国六(四)15），`floor=None` 保留原值 | 草稿：不处理 |
| 7 | 满额或超额使用的 CCF | `limit <= drawn` 时为 NaN，文档给出 `ead / limit` 的写法 | 增加 `fully_drawn="limit_factor"`，按尚未定稿的 EBA CP §61(b) 计算 |
| 8 | 模拟数据 | `make_defaults` 和 `make_recoveries` 两个函数，与 `make_applications`、`make_transactions` 一致 | 一个函数返回两张表的元组 |
