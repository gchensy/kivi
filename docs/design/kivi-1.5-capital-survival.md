# kivi 1.5 实施计划：监管资本与生存分析

2026-11-24 起草 · 2026-11-24 审定：第 6 节的 7 项全部采用推荐

## 1. 目标与范围

1.5 的内容已由 `docs/design/kivi-1.4-risk-parameters.md` 第 4、5 节设计，2026-11-23 审定（该稿第 10 节第 6–11 项）：

- **新模块 `kivi.capital`：** IRB 风险权重函数、Vasicek 单因子模型、Pluto–Tasche 最审慎估计。
- **`kivi.lifetime.survival_table`：** Kaplan–Meier 和 Aalen–Johansen 累计发生率，支持左截断。
- **`make_loan_panel(prepayment=0.0)`：** 模拟提前结清，默认输出不变。

该稿规定 1.5 开工前要再审定 API，并在写代码之前把公式与监管原文逐条核对。本稿做三件事：

1. 记下核对结果和出处；
2. 据此修订 API；
3. 给出会话分工、测试依据和待你决定的事项。

运行时仍只依赖 NumPy 和 pandas。

## 2. 原文核对结果

2026-11-24 核对，依据的文本如下。

- **巴塞尔：**
  - 现行框架（2023 年 1 月 1 日起生效）CRE30–CRE32 各章的 PDF：
    - https://www.bis.org/committees/bcbs/basel-framework/81395/chapter.pdf （CRE31）
    - …/81396/ （CRE32）
    - …/81394/ （CRE30）
  - 公式在 PDF 中是图片，已逐页渲染后转录。
  - 与 d424（2017 年 12 月，第 52–54 段）和 bcbs128（2006 年巴塞尔 II，第 272–273、328–330 段）的文字版一致。
- **中国：** 《商业银行资本管理办法》（国家金融监督管理总局令 2023 年第 4 号）。
  - 政府网站从这里无法连接（TLS 重置），所以：
    - 正文取自东方财富转载的总局排版 PDF；
    - 附件 4、6、8、21 取自 gov.cn 官方 .docx 在 Wayback 2025-04-18 的快照。
  - 附件 6 的公式是 Equation 3.0 对象，已渲染后转录。周围的文字按原文引用。
- **ECB：** "Instructions for reporting the validation results of internal models"，最新版仍是 2019 年 2 月。
  - 监管页面（2026-04-16 更新）只链接这一版。
  - 2021 年只重发了报告模板。
  - 1.4 的验证检验公式无需改动。

### 2.1 公式：与 #79 草稿一致

两地的公式与草稿相同：

- 公司、主权、银行的相关系数（CRE31.5；附件 6 一(一)1）；
- 期限调整 b = (0.11852 − 0.05478·ln PD)²；
- 资本要求 K 的公式与 99.9% 置信水平；
- 零售三类（CRE31.13–31.16；附件 6 一(一)4）：个人住房抵押 0.15，合格循环零售 0.04，其他零售 0.03/0.16（指数 35），零售不做期限调整；
- RWA = 12.5·K·EAD；
- 已违约敞口 K = max(0, LGD − ELBE)（CRE31.3；附件 6 二，原文为 K＝Max[0,（LGD－BEEL）]）。

复算（SciPy）：

| 情形 | R | K | 风险权重 |
| --- | --- | --- | --- |
| 公司，PD 1%，LGD 45%，M 2.5 | 0.192784 | 0.073853 | **92.3168%**（草稿的 92.32% 正确） |
| 其他零售，PD 1%，LGD 45% | 0.121609 | 0.036618 | 45.7727% |
| 个人住房抵押，PD 1%，LGD 25% | 0.15 | 0.025066 | 31.3327% |
| 合格循环零售，PD 1%，LGD 80% | 0.04 | 0.024497 | 30.6207% |
| 中小企业，S = 500 万欧元 | | | 72.3947% |
| 金融机构 ×1.25 | | | 117.9494% |

巴塞尔 II 附件 5 的 152 个示例风险权重全部能复现到 0.01 个百分点以内，覆盖公司、中小企业、住房抵押、其他零售和循环零售，PD 从 0.03% 到 20%，可直接用作测试数据。

### 2.2 与草稿不同或草稿未写的地方

| 项目 | 巴塞尔（现行） | 中国 2023 | 对 API 的影响 |
| --- | --- | --- | --- |
| 1.06 调整系数 | 已取消：d424 引言表格脚注 3 写明"will no longer apply"。2019 版 CRE30.4 有，2023 版没有 | 没有：附件 6 一(四) 只有 RWA = K×12.5×EAD。2012 版也没有 | 默认 `scaling=1.0` 正确（决定 5） |
| 中小企业规模调整 | −0.04·(1 − (S − 5)/45)，S 为年销售额（百万欧元），夹在 [5, 50]（CRE31.8）；监管可允许用总资产代替（CRE31.9） | −0.04×(1 − (S − 3)/27)，S 为近 3 年营业收入的算术平均（千万元人民币），低于 3 按 3 计（附件 6）；上限 30 来自中小企业定义"不超过 3 亿元"（附件 4 二(三)） | `turnover_range` 关键字已能表达；指南写明两地的区间和单位 |
| 金融机构 ×1.25 | 总资产不低于 1000 亿美元的受监管金融机构，以及任何规模的无监管金融机构（CRE31.7）；印出的公式不含中小企业项 | 金融机构风险暴露一律 ×1.25，只有非系统重要性银行用公司公式（附件 6 一(一)2）；没有资产门槛 | 是否适用由调用者判断，`financial` 仍为布尔值（决定 4）；与中小企业调整同时出现时怎么办见决定 3 |
| 高波动商业房地产（HVCRE） | R = 0.12·w + 0.30·(1 − w)（CRE31.11） | 同（附件 8 一(二)） | 草稿未包括，建议增加（决定 1） |
| 主权的极小 PD | 主权没有 PD 下限。巴塞尔 II 脚注 72、d424 脚注 13 和 2019 版 CRE31.5 规定"负的资本要求按零计"，2023 版删去 | 未见相关规定 | 当 1 − 1.5·b ≤ 0，即 PD ≤ 2.927e-6 时公式失效：PD 低于约 1e-5 后风险权重反而上升（PD 3e-6 时为 46%，2.95e-6 时为 145%），低于 2.927e-6 后变为负值。需要处理（决定 2） |
| PD 下限 | 主权以外不低于 0.05%（CRE32.4）；零售一般 0.05%，非交易型循环零售 0.1%（CRE32.58）；巴塞尔 II 为 0.03% | 公司和金融机构 0.05%，主权无下限（第九十一条）；零售 0.05%，一般循环零售 0.1%；2012 版为 0.03% | 由调用者用 `np.maximum` 施加，指南列出 |
| LGD 下限（高级法） | 公司无担保 25%；有担保：金融质押 0%、应收账款和房地产 10%、其他实物 15%（CRE32.16–17）；住房抵押 5%，循环零售 50%，其他零售无担保 30%（CRE32.58–59） | 同巴塞尔，但个人住房抵押为 **10%**（第九十二条） | 指南列出 |
| 初级法监管 LGD | 主权、银行、金融机构的优先无担保 45%，其他公司 40%（巴塞尔 II 为 45%），次级 75%（CRE32.6–7） | 同（45% / 40% / 75%） | 指南列出 |
| 期限 M | 初级法 2.5 年，回购类 0.5 年（CRE32.44）；一般夹在 [1, 5] 年（CRE32.46）；短期例外可低到 1 天（CRE32.51–54）；无其他规定时 2.5 年（CRE32.55） | 初级法 2.5 年，回购类 0.5 年；高级法为内部估计与 1 年的较大值、不超过 5 年，中小企业可用 2.5 年（第九十四条） | 维持草稿：不自动截断，默认 2.5 |
| 高级法的适用范围 | 合并收入超过 5 亿欧元的公司、银行和其他金融机构不得用高级法（CRE30.34–35） | 金融机构，以及年营业收入超过 30 亿元的公司不得用高级法（第八十九条） | 指南说明 |
| 输出下限 | 72.5%（另章） | RWA = max(RWAa, RWAs×72.5%)（附件 21 四） | 组合层面的规则，需要标准法 RWA，1.5 不做，指南提及（决定 7） |

## 3. API：在 #79 第 4、5 节基础上的修订

只列出变化，其余不变。

### 3.1 `kivi.capital`

```python
asset_correlation(probability, *, asset_class, turnover=None,
                  turnover_range=(5.0, 50.0), financial=False)
capital_requirement(probability, lgd, *, asset_class, maturity=None, turnover=None,
                    turnover_range=(5.0, 50.0), financial=False, correlation=None,
                    level=0.999, scaling=1.0)
conditional_pd(probability, *, correlation, factor=None, level=None)
systematic_factor(default_rate, probability, *, correlation)
vasicek_fit(default_rate)
most_prudent_pd(event, total, *, level=0.95, correlation=0.0)
```

- **`asset_class` 增加 `"hvcre"`（决定 1）。** 取值为 `"corporate"`、`"hvcre"`、`"mortgage"`、`"revolving"`、`"other_retail"`。`"hvcre"` 用 0.12/0.30 并做期限调整，不接受 `turnover` 和 `financial`。
- **极小 PD（决定 2）：** 非零售类别在 1 − 1.5·b ≤ 0（PD ≤ 2.927e-6）时报错。错误信息给出界限，并提示先施加 PD 下限。docstring 的 Notes 写明 PD 低于约 1e-5 时风险权重随 PD 下降而上升，原因在期限调整。
- **金融机构与中小企业（决定 3）：** `financial=True` 与小于上限的有限 `turnover` 同时给出时报错。两地印出的金融机构公式都不含中小企业项，`turnover=np.inf` 或不传仍然可以。
- **`scaling`（决定 5）：** 保留，默认 1.0。docstring 说明两地现行规则都没有 1.06，只有巴塞尔 II 有。
- **违约敞口：** 维持草稿，PD = 1 时报错，提示按 CRE31.3 / 附件 6 二计算 max(0, LGD − ELBE)。

### 3.2 `kivi._stats`

维持草稿：

- 数组版 `normal_quantile`，标量调用照旧；
- `beta_quantile`；
- `bivariate_normal_cdf`。

新增函数的测试放在新文件 `tests/test_stats.py`，这样不与其他测试文件冲突。

### 3.3 `kivi.lifetime.survival_table` 与 `make_loan_panel`

维持草稿：

```python
survival_table(duration, event, *, competing=None, entry=None, weights=None)
```

返回的表以 `period` 为索引，列为 `at_risk`、`event`、`competing`（只在给出 `competing` 时有）、`censored`、`hazard`、`survival` 和 `cumulative_pd`。另有 `make_loan_panel(prepayment=0.0)`。

参数名 `event` 见决定 6。

## 4. 测试与参考

| 内容 | 对照 |
| --- | --- |
| `asset_correlation`、`capital_requirement` | 巴塞尔 II 附件 5 的 152 个风险权重（硬编码，误差 0.01 个百分点）；用 `scipy.stats.norm` 独立写出的公式在 PD、LGD、M、S 和各资产类别的网格上比对到 1e-12；第 2.1 节表中的固定值 |
| 极小 PD、金融机构与中小企业 | 边界两侧的报错与取值 |
| `conditional_pd`、`systematic_factor`、`vasicek_fit`、`most_prudent_pd` | 同 #79 第 9.1 节：互为逆运算；模拟恢复参数；原论文算例 0.29%、0.33%、0.76%；`scipy.integrate.quad` |
| `_stats` 新函数 | SciPy 的 `norm.ppf`、`beta.ppf`、`multivariate_normal.cdf` |
| `survival_table` | statsmodels `SurvfuncRight`（含 `entry` 和 `freq_weights`）；`CumIncidenceRight` 没有 `entry`，左截断下的累计发生率手算；完整观察的批次与 `vintage_table` 一致 |

功能 PR 的交叉验证先放在各自的测试文件里，用 `pytest.importorskip`；集成 PR 再移入 `tests/test_reference.py`。

## 5. 会话与流程

| 会话 | PR | 文件 |
| --- | --- | --- |
| `capital` | 1. `_stats` 小 PR | `src/kivi/_stats.py`、`tests/test_stats.py`（新） |
| | 2. `kivi.capital`，在 1 合并后开 | `src/kivi/capital.py`（新）、`tests/test_capital.py`（新）、`docs/user_guide/capital.rst`（新）、`docs/api/capital.rst`（新）、发布说明片段 |
| `survival` | 一个 PR，与 `capital` 并行 | `src/kivi/lifetime.py`、`src/kivi/datasets/_loan_panel.py`、`tests/test_lifetime.py`、`tests/test_datasets.py`、`lifetime.rst` 和 `datasets.rst` 的相关小节、发布说明片段 |

两个会话的文件互不重叠，`kivi.po` 用 `--merge-po` 解决冲突。

流程：

1. 本稿审定：2026-11-24 已确认，第 6 节的 7 项全部采用推荐。
2. 两个会话开工。我逐个评审，按你的指示合并。
3. 集成 PR 由我来写：
    - README、首页和指南索引（`capital` 指南放在 `lifetime` 之前或之后，由集成 PR 定）；
    - `compatibility.rst`：`capital` 的 Series 名和 `survival_table` 的列；
    - glossary 与 conventions：新参数名 `asset_class`、`duration`、`event`、`competing`、`entry`，以及资产相关系数、Vasicek、Pluto–Tasche、Kaplan–Meier、Aalen–Johansen、左截断等术语；
    - 测试登记表，以及交叉验证移入 `test_reference.py`；
    - 图：资本曲线、生存曲线；
    - CLAUDE.md；
    - 若 `capital` 先于 `survival` 合并，多情景前瞻性调整的示例由 `survival` 会话写进 lifetime 指南，否则放进集成 PR。
4. 发布 1.5.0，需要你的确认。

1.6（`kivi.lgd`）和 1.7（`kivi.rules`、收益）在 1.5 发布后按同样的流程开工：先核对原文，例如 EBA 关于清收 LGD 折现率的指引和 EMP 的参考值，再审定 API。

## 6. 决定

2026-11-24，以下 7 项全部采用推荐方案。

| # | 问题 | 决定 | 未采用的选项 |
| --- | --- | --- | --- |
| 1 | 高波动商业房地产 | 增加 `asset_class="hvcre"`（0.12/0.30，做期限调整）。两地都有，只多一个分支 | 不做，由调用者用 `correlation=` 自行传入 |
| 2 | 非零售类别的极小 PD | 在 1 − 1.5·b ≤ 0（PD ≤ 2.927e-6）时报错并给出界限；docstring 说明 PD 低于约 1e-5 时期限调整使风险权重回升 | 按巴塞尔 II 脚注把负的 K 记为 0（界限附近的异常大值仍在）；不处理 |
| 3 | 金融机构与中小企业同时出现 | 报错：两地印出的金融机构公式都不含中小企业项 | 先做中小企业调整再 ×1.25（草稿的做法，超出原文）；忽略 `turnover` |
| 4 | `financial` 的含义 | 保持布尔值，是否适用 ×1.25 由调用者按辖区规则判断，指南列出巴塞尔（1000 亿美元门槛加无监管机构）和中国（除非系统重要性银行外的金融机构）的规则 | 增加 `total_assets` 等参数，在函数内判断 |
| 5 | `scaling` | 保留关键字，默认 1.0，注明只有巴塞尔 II 有 1.06 | 删去 |
| 6 | `survival_table` 的事件参数名 | `event`：生存分析的通用写法，kivi 中 `event` 也一直指目标为 1 的观测或其计数 | `target`，与其他函数的二元标签一致 |
| 7 | 输出下限（72.5%） | 1.5 不做，指南提及：它是组合层面的规则，需要标准法 RWA | 增加函数 |
