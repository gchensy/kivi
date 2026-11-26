# kivi 1.8 设计：一年违约率、长期平均违约率与对准中心趋势的校准

2026-11-25 起草 · 待审定：第 6 节的 9 项

## 1. 目标与范围

`docs/design/kivi-1.8-roadmap.md` 第 3.3 节（2026-11-25 审定）给 `default-rates` 会话排了三个暂名的函数：

- **`kivi.vintage.default_rate_table`：** 用队列法从月度面板计算各观察起点、各等级的一年违约率；
- **`kivi.calibration.central_tendency_shift`：** 给所有 log odds 加同一个常数，使平均 PD 等于中心趋势；
- **`kivi.calibration.quasi_moment_matching`：** 拟合 logistic PD 曲线，使平均 PD 等于中心趋势、AR 等于目标值。

长期平均违约率不设函数，在指南中用一行 pandas 计算。第 2.3 节起，第三个函数按决定 1 的推荐称为 `quasi_moment_fit`。

与 1.5–1.7 一样，开工前先核对原文，再审定 API。本稿做三件事：

1. 记下核对结果和出处，并说明 kivi 已有哪些部分（第 2 节）；
2. 据此修订 API（第 3 节）；
3. 给出测试依据、流程和待你决定的事项（第 4–6 节）。

1.8 不需要新的运行时依赖：三个函数都只用 NumPy 和 pandas。

## 2. 原文核对结果

2026-11-25 核对，依据的文本如下。

- **EBA/GL/2017/16**（PD 估计、LGD 估计和违约敞口处理的指引，2017 年 11 月 20 日的最终报告），取自 EBA 网站的 PDF：
  - 第 2.4 节的定义；
  - 第 5.3 节第 70–99 段；
  - 背景说明中关于一年违约率、观察平均、长期平均和校准的部分（报告第 21–26 页，含表 1 “评级理念与校准理念”）。
- **CRR 第 4(1)(78) 条**，一年违约率的定义。EBA 73 援引它并逐项展开，本稿按 EBA 73 实现。EUR-Lex 拒绝自动访问，原文未逐字核对。
- **巴塞尔框架 CRE36.77–36.82**（2023 年 1 月 1 日起生效的版本），取自 BIS 的 PDF。
- **ECB**，“Instructions for reporting the validation results of internal models – IRB Pillar I models for credit risk”（2019 年 2 月）第 2.3 节 (a)–(d)、第 2.5.1 节 (b)–(h)、第 2.5.3.1 节。与 1.4、1.6 依据的是同一版本。
- **D. Tasche**，“Estimating discriminatory power and PD curves when the number of defaults is small”，Lloyds Banking Group 工作论文，arXiv:0905.3928v2（2010 年 3 月 5 日）。草稿说的“2009 年提出”指的就是它：第 3.2 节定义 3.14 和命题 3.15，第 5.2 节式 (5.13b)–(5.17) 和例 5.4。
- **D. Tasche**，“The art of probability-of-default curve calibration”，*Journal of Credit Risk* 9(4)，2013，63–103；依据 arXiv:1212.3716v6（2013 年 11 月 26 日）。用到的是：
  - 第 2–3 节，表 2–5；
  - 第 4.2–4.3 节，表 7；
  - 第 5 节，表 12；
  - 附录 A。
- **未核对：**
  - 《商业银行资本管理办法》附件 5 关于 PD 的部分。指南只引用 EBA、巴塞尔和 ECB，需要中国对照时在集成 PR 之前另行核对。
  - 违约定义指引 EBA/GL/2016/07 关于考察期的段落。下文提到它时没有给段落号。

### 2.1 一年违约率

| 项目 | EBA / CRR / 巴塞尔 | ECB 2019 | 对 API 的影响 |
| --- | --- | --- | --- |
| 定义 | 分母是一年观察期开始时有任何信用债务（表内或表外，含担保）的未违约债务人数；分子是其中在观察期内至少发生一次违约事件的债务人数（EBA 73，展开 CRR 4(1)(78)） | 分母是期初未违约、有正敞口或已承诺未提用额度的客户；分子是其中在观察期内至少发生一次违约（开始）事件的客户，迁移到其他模型或资本计算方法之后的违约也算（2.3(b)(c)） | 队列法：起点那个月未违约的贷款进入分母，此后 `horizon` 个月内任一月违约即计入分子 |
| 计数单位 | 债务人（EBA 73） | 客户可以是债务人，适用时也可以是债项（2.3(b)） | `loan` 传债务人编号或债项编号，由违约定义所在的层级决定，docstring 说明 |
| 分组 | 按观察期开始时的等级或池；计入推翻（override），不反映信用风险缓释的替代效应和事后保守调整（EBA 74）。背景说明：来自无资金信用保护提供者的评级转移，按转移前的等级计 | 评级取信用风险缓释替代之前、评级转移之后的（2.3(b)）；评级被转移的客户不计入 N（2.5.1(g)） | 只用起点那一行的 `grade`，以后的等级不影响结果。起点评级怎样确定是调用者的数据步骤，docstring 列出两处的规则 |
| 缺失评级 | 期初没有评级、但在模型适用范围内的债务人（“missing ratings”）也要单独计算一年违约率（EBA 75） | 缺失评级属于流程缺陷造成的数据剔除，不计入 N（2.5.1(b)(c)(g)） | 缺失的 `grade` 自成一行，与 `_counts` 相同，排在最后：EBA 75 直接可用，ECB 的口径去掉这一行即可（决定 5） |
| 期间退出与迁移 | 迁移到其他等级、池、模型、评级体系或资本计算方法的，以及债务在观察期内被出售、核销、偿清或以其他方式结清的，都仍然计入分母，违约的计入分子；机构须分析这些情形是否造成偏差，必要时调整并加 MoC（EBA 76）。背景说明给出的理由是一年违约率的口径要统一、明确 | 迁移后的违约也计入（2.3(c)）；迁移矩阵为终止业务关系的客户单列一栏，他们属于 N（2.5.1(h)(iv)） | **退出不删失：** 窗口内没有违约记录就离开面板的贷款计为未违约，这与 `lifetime.survival_table` 的 Kaplan–Meier 删失不同（第 2.5 节）。新增 `exited` 列，供 EBA 76 和 80(a) 要求的偏差分析使用（决定 4） |
| 多次违约 | 一个观察期内多次违约只计一次（EBA 77） | 只计一次（2.3(c)） | 窗口内至少有一个违约月，就计一次 |
| 技术性违约与错误违约 | 不符合违约定义而被错误记为违约的，不计入分子（EBA 71(a)） | 技术性违约不计入（2.3(d)） | 由调用者从 `defaulted` 中去掉，docstring 说明 |
| 其他剔除 | 错误归入该模型的债务人，从分子和分母中剔除（EBA 71(b)），剔除要记录（EBA 72） | 剔除评级过期（期初前 12 个月以上）、财务报表过期（24 个月以上）和评级被转移的客户（2.5.1(d)(f)(g)） | 调用者在传入之前筛选面板 |
| 观察频率与窗口 | 至少按季度评估历史观察期内的一年违约率（EBA 78）。重叠或不重叠的一年窗口二选一，选择前须分析：(a) 观察不到的短期合同和已终止合同造成的偏差；(b) 计算日的选择；(c) 重叠窗口对重叠期的隐含加权；(d) 季节性（EBA 80） | 一个统一的一年观察期，通常与机构内部验证的观察期相同（2.3(a)） | `starts` 默认取每个季末，即 EBA 78 的最低频率。不重叠窗口每年传一个月，ECB 的报告传一个起点（决定 2） |
| 加权 | 各等级的 PD 基于按债务人数简单平均（count weighted）的历史平均一年违约率，不允许 EAD 加权（CRE36.78） | 检验按客户数 N 和违约数 D（2.5.3），另外报告原始敞口 | `weights` 只作频数权重，如抽样面板；docstring 写明 PD 校准不得按敞口加权 |

### 2.2 观察平均违约率与长期平均违约率

| 项目 | 原文 | 对 API 的影响 |
| --- | --- | --- |
| 计算的层面 | 每个等级或池、模型覆盖的敞口类型、每个校准分段，都要计算观察平均违约率（EBA 79） | 指南给出按等级（`groupby(level="grade")`）和按组合（先按起点汇总计数）的两种写法 |
| 观察平均 | 所有一年违约率的算术平均。零售模型在近期数据更能预测损失时，可以用加权平均（EBA 81，CRR 180(2)(e)） | 指南写法：`table["event_rate"].groupby(level="grade").mean()`；零售的加权平均写成 `numpy.average` |
| 观察期长度 | 至少最近 5 年。为了反映违约率可能的变化范围，需要时加入更早的观察（EBA 82）。巴塞尔：至少一个数据源覆盖 5 年以上；有更长的相关数据时必须使用；数据应包含好年份和坏年份的代表性组合（CRE36.79，零售 CRE36.82） | 不由函数检查，指南说明 |
| 代表性 | 评估以下三项：(a) 一年违约率的变异；(b) 按经济指标判断，坏年份是否存在、缺失或过多；(c) 经济、法律或业务环境的重大变化（EBA 83） | 机构的判断，指南说明 |
| 长期平均 | 观察期有代表性时，长期平均就是观察平均（EBA 84）。不具代表性时：缺少坏年份必须调整；坏年份过多时可以向下调整，前提是经济指标与违约率显著相关（EBA 85） | 调整属于机构的判断，不设函数（决定 9） |
| 向下调整的基准 | 调整后低于全部观察平均时，要与以下两者中的较大者比较：最近 5 年的观察平均、全部观察平均。低于这个较大者时须专门论证（EBA 86） | 指南用两行写出这个基准 |

### 2.3 对准长期平均的校准

- **定义（EBA 第 2.4 节）：** PD 校准，是指把排序或分池方法用于校准样本时，得到的 PD 估计与相应层面的长期平均违约率一致。
- **检验（EBA 87）：** 按等级和按校准分段做定量的校准检验，再加定性分析，如每个等级的最少债务人数、避免过度集中。
- **校准样本（EBA 88）：** 在债务人和交易特征上与申请组合可比，同时代表违约率可能的变化范围。
- **顺序（EBA 89）：** 在计入推翻之后、加 MoC 和 PD 下限之前校准。
- **层面（EBA 92）：** 两种层面二选一，另做另一层面的检验：
  - (a) 按等级的长期平均，另做校准分段层面的检验；
  - (b) 按校准分段的长期平均，另做等级层面的检验。
- **限制（EBA 96）：** 不得用连续 PD 或平滑违约率来弥补数据不足或区分能力的缺陷，也不得用来降低资本要求。
- **排序（EBA 99）：** 校准不应改变校准分段内的排序，等级内部除外。
- **校准理念（背景说明表 1）：**
  - **TTC B：** 在较近的时点评估 PD，再把它们调整到平均值等于校准分段的长期平均违约率（“Adjust PD on portfolio level to LRA DR; use of current average PD estimates”），其中所选时点须与当前组合可比、代表违约率的变化范围；
  - **TTC C：** 同样在组合层面调整，但校准样本覆盖整个历史观察期的多个时点。

**对 API 的影响：**

- `central_tendency_shift` 正是 TTC B 和 TTC C 在组合层面的调整：所有 log odds 平移同一个常数，所以排序不变（EBA 99）。指南说明应在 MoC 和下限之前做（EBA 89）。
- 按等级校准（EBA 92(a)）有两种做法：一是 `fit_grade_pd`；二是在等级上用 `quasi_moment_fit`，即 Tasche 平滑估计期 PD 曲线的做法。
- 按等级的检验（EBA 87(a)）是 `default_rate_table` 各行配上 `jeffreys_test`。

### 2.4 Tasche 的两种方法

#### Quasi moment matching

- **出处：** Tasche（2009）第 5.2 节，式 (5.16a)、(5.16b) 和例 5.4 的式 (5.17)。称为 “quasi”，是因为 AUC\* 不是条件 PD 的真正矩。不用 PD 的方差作第二个方程，是因为它随违约者的比例变化，而区分能力随时间的变化较小。2013 年的综述有两处用到它：
  - 附录 A：平滑估计期的 PD 曲线；
  - 第 4.2.2 节：预测期的“不变 AR”做法。
- **曲线：**
  - 2009 年例 5.4 用原始分数的 logit 曲线：P[D | S = s] = 1 / (1 + exp(α + β s))（式 5.14）；
  - 2009 年式 (5.13b) 和 2013 年式 (A.8) 用 robust logit：把 s 换成 Φ⁻¹(F_N(s))，其中 F_N 是存活者的分数分布函数；
  - 2013 年式 (A.9)：用 F_N 与其左连续版本的平均 F̃_N，避免离散分布出现 F_N = 1；
  - 2013 年第 4.2.2 节：预测期没有违约结果，用无条件分布近似 F̃_N。
- **两个方程：**
  - 平均 PD 等于目标的无条件 PD（式 5.16a）；
  - 由 PD 曲线隐含的 AUC\* 或 AR 等于目标值（式 5.16b；按等级的写法见 2013 年式 A.11b）。
- **中心趋势的含义（2013 年第 4 节第 17 页）：** 预测期的无条件 PD 可以是最佳估计，可以是压力测试用的悲观估计，也可以是 TTC 校准用的长期估计。
- **AR 的定义（2009 年定义 3.14、命题 3.15，式 3.28a–b；2013 年式 A.11a 沿用）：**
  - AUC\* = P[S_D < S_N] + ½ P[S_D = S_N]；AR\* = 2 AUC\* − 1 = P[S_D < S_N] − P[S_D > S_N]；
  - 并列按一半计，所以对离散分布也满足“没有区分能力为 0、完全区分为 1”；
  - kivi 的 `metrics.roc_auc` 和 `gini` 同样按一半计并列（梯形面积），区别只是方向：Tasche 的分数越低越危险，kivi 的分数随风险上升。
- **并列的细节：**
  - 2013 年式 (A.11b) 按等级计算，含同一等级内配对的一半，即 ½ Σ PD_x (1 − PD_x) π_x²；
  - 2009 年的样本式 (5.17) 含每个观测与自身配对的一半；但对分数相同的不同观测，按排序的先后计，有并列时与式 (5.16b) 不完全一致；
  - kivi 按式 (5.16b) 和 (A.11b) 实现：所有并列（含自身配对）都按一半计。这等于把每个观测拆成权重为 w·PD 的事件和 w·(1 − PD) 的非事件，再计算 `gini`。
- **求解：** 2013 年附录 A 用等方差二元正态模型的式 (A.6)、(A.2c)、(A.2d) 给出初值，再数值求解二元非线性方程组。

#### 对准中心趋势

2013 年第 4.2 节讨论情形 1：已知预测期的无条件评级分布和无条件 PD p₁。Tasche 比较了四种做法：

| 做法 | 公式 | Tasche 的结论 | kivi |
| --- | --- | --- | --- |
| 不变违约分布 (4.2) | PD₁(x) = p₁ Pr₀[x \| D] / Pr₁[x]（式 4.6b） | 须满足式 (4.6a) 才是合法的模型 | 不做 |
| 不变 AR (4.3) | 在预测期的分布上用 QMM，目标为 p₁ 和 AR₀ | 表 7 中与缩放似然比同为最好（2011 年 p 值最高）；在 Moody's 回测中最差（表 12，平均秩 2.05，秩越高越好） | `quasi_moment_fit` 可以实现 |
| 按比例缩放 PD (4.4) | PD₁ = c·PD₀，c = p₁ / Σ PD₀ π₁（式 4.7a） | 可能超过 1，并把估计期的无条件 PD 带进预测（“contamination”），理论上有问题（平均秩 2.61） | 不设函数，指南提一句 |
| 按比例缩放似然比 (4.5) | λ₁ = c·λ₀，c 由 Σ π₁(x) / (p₁ + (1 − p₁) c λ₀(x)) = 1 唯一确定，PD₁(x) = p₁ / (p₁ + (1 − p₁) c λ₀(x))（式 4.8a、4.8d） | 理论上合理，表现最好（表 7 中与不变 AR 同为最好，2010 年 p 值最高；回测平均秩 2.76，四种中最好） | `central_tendency_shift` |

**缩放似然比就是平移 log odds。** 由式 (3.9b)，λ₀(x) 是等级 x 存活的 odds 与无条件存活 odds 之比，所以 λ₁ = c·λ₀ 等于给每个等级违约的 log odds 加同一个常数 Δ。式 (4.8a) 又等价于“预测期分布上的平均 PD 等于 p₁”。两者合起来，正是草稿的 `central_tendency_shift`。Tasche 证明这个常数总是存在且唯一（第 22 页）：平均 PD 随常数严格单调，两端的极限分别为 0 和 1/p₁ > 1。kivi 的求解用的是同一个单调性。

**情形 2：不知道预测期的评级分布（第 4.3 节）。** 这时只能假设似然比不变（式 4.10），PD₁(x) = p₁ / (p₁ + (1 − p₁) λ₀(x))（式 4.11）。这就是 Elkan 的 “change in base rate” 定理，也就是 kivi 的 `prior_correction(probability, sample_rate=p₀, population_rate=p₁)`。

**分工：**

- `prior_correction` 用于两种情形：修正开发样本的抽样；或者新模型还没有覆盖当前组合，不知道当前的评级分布；
- `central_tendency_shift` 用于已知当前组合或校准样本的情形，即 EBA 的 TTC B 和 TTC C。

#### 参考值

Tasche（2013）表 2 给出了 S&P 2009–2011 年 17 个等级的评级数和违约数，下文都以它为输入。复现在临时环境里用 SciPy 完成，脚本不进仓库：

- **表 4：** 2009 年违约率 234/5860 = 3.993%。用 `kivi.metrics.gini` 按计数加权计算表 2，AR = 0.827102，与表 4 的 82.7% 一致。
- **表 5（QMM，robust logit）：**
  - **做法：** 按式 (3.5c) 由 2009 年的存活者计数得到存活分布，取 t = Φ⁻¹(F̃_N)；以评级数为权重，目标为未经四舍五入的 234/5860 和 0.827102。
  - **解：** α = 5.20481，β = 2.16127（Tasche 的方向）。
  - **吻合程度：** 17 个格子中 15 个四舍五入到表中的 3 位小数（%）后完全相同。BB（1.5186 对 1.518）和 CCC-C（48.3537 对 48.355）差末位，最大绝对差 0.0013 个百分点。
  - **输入的口径：** 改用表 3 四舍五入的百分比，或四舍五入的 3.99% 和 82.7% 作目标，都有 3–9 个格子不同。可见 Tasche 用的是计数和未四舍五入的目标，剩下的差异与他的求解精度相符。
- **表 7，“不变 AR”列：** 在 2010、2011 年的无条件分布上，以其 F̃ 作变换，目标为 p₁ 和 AR₀（第 4.2.2 节）：
  - 2010 年 17 个格子中 15 个相同，最大差 0.00017 个百分点；
  - 2011 年 13 个相同，最大差 0.00042 个百分点。
- **表 7，“缩放似然比”列：** 对表 5 印出的 PD 平移 log odds，以 2010、2011 年的评级数为权重，中心趋势为 63/5522 和 44/5847：
  - 最大差 0.00009（2010 年）和 0.0003（2011 年）个百分点；
  - 改用上面未四舍五入的 QMM 结果作输入，最大差为 0.00048 和 0.00021 个百分点。

**AR 随斜率单调上升。** 平均 PD 固定时，logistic 曲线隐含的 AR 随斜率上升。Tasche 没有给出证明，本稿做了数值检验：

- **样本：** 300 组随机样本，分数为正态、6 个取值的大量并列和重尾三种，权重为单位权重或整数权重，中心趋势取 1%–60%；
- **网格：** 每组样本取 400 个斜率；
- **结果：** 相对下降不超过 1.3e-14，即只有舍入误差。

因此第 3.3 节把二元方程组拆成两个一维问题求解。

### 2.5 kivi 已有的部分

| 函数 | 与本稿的关系 |
| --- | --- |
| `calibration.prior_correction`、`corrected_intercept` | King–Zeng 的修正，即 Tasche 式 (4.11) 的“似然比不变”。odds 的倍数 ω 由两个事件率给定，当前组合上的平均 PD 不一定等于目标。`central_tendency_shift` 反过来求 ω = e^Δ，使平均 PD 恰好等于目标。两者都保留，指南说明各自适用的情形（第 2.4 节） |
| `calibration.calibration_table` | 单个起点的 `default_rate_table` 各行，等于对“起点等级、窗口内违约”调用 `calibration_table` 的结果。新函数做的是从面板构造多个起点的队列，列名和计数规则沿用它 |
| `calibration.jeffreys_test`、`binomial_test` | 对 `default_rate_table` 的每一行，或 ECB 报告的那一个起点做检验 |
| `calibration.fit_grade_pd`、`master_scale`、`assign_grades` | 按等级校准（EBA 92(a)）的现有工具。`fit_grade_pd(event, total)` 接受计数，把各起点的计数相加，得到的是合并违约率，而不是 EBA 81 的算术平均，指南说明两者的差别 |
| `calibration.platt_fit`、`platt_predict` | 与 QMM 是同一条 logistic 曲线。QMM 返回的参数直接交给 `platt_predict` |
| `vintage.vintage_table` | 按放款队列和账龄统计的累计坏账率。分母包含期间退出的贷款，与 EBA 76 的原则相同；但它的队列是放款月份，没有等级，也没有“起点时未违约”的条件，算不出按观察起点和等级的一年违约率 |
| `vintage.delinquency_panel` 的 `status` | 月末逾期桶 `current`、`1-29`、…、`90+`。`status == "90+"` 只是违约定义中逾期 90 天的部分（CRR 178(1)(b)），不含“不太可能偿付”、重要性门槛和 EBA/GL/2016/07 的至少 3 个月考察期。月末快照还会漏掉月内违约又在月末前治愈的情形。docstring 要求 `defaulted` 表示“当月任何时候处于违约，含考察期” |
| `vintage.transition_matrix`、`migration_test` | 只看期末的状态（ECB 2.5.1(h) 脚注 26）。12 个月迁移矩阵的违约列漏掉期间违约又治愈的客户，与 EBA 73(b) “至少一次违约事件”不同 |
| `lifetime.survival_table` | Kaplan–Meier 把退出当作删失，1 − S(12) 高于队列违约率。在 `make_loan_panel(rng=0)` 上，2024-12 起点的 `current` 池有 2146 笔贷款，其中 19% 在窗口内退出：队列违约率 3.40%，Kaplan–Meier 约为 3.9%。IRB 的一年违约率是队列率（EBA 73、76）；`survival_table` 仍然用于 IFRS 9 的整个存续期 |
| `capital.vasicek_fit` | 接受每期一个违约率。把组合层面、每年一个起点（不重叠窗口）的违约率序列交给它；重叠窗口的序列自相关，会使方差估计失真，指南说明 |
| `datasets.make_loan_panel` | 没有等级。指南把起点时的逾期状态（`current`、`1-29`、`30-59`、`60-89`）当作零售的分池，以 `90+` 作违约。面板覆盖 2024-02 至 2026-01，12 个月窗口完整的季末起点有 2024-03 至 2024-12 四个；组合违约率依次为 6.83%、6.26%、5.64%、5.20%。数值来自暴力原型，以实现为准 |

## 3. API：在 1.8 路线图第 3.3 节基础上的修订

```python
# kivi.vintage
default_rate_table(loan, month, grade, defaulted, *, horizon=12, starts=None,
                   probability=None, weights=None)
# -> DataFrame，二级索引 (start, grade)：total、event、exited、event_rate，
#    给出 probability 时再加 mean_probability

# kivi.calibration
central_tendency_shift(probability, *, central_tendency, weights=None)
# -> float：log odds 的平移量
quasi_moment_fit(score, *, central_tendency, accuracy_ratio, weights=None)
# -> (slope, intercept)，交给 platt_predict
```

### 3.1 `default_rate_table`

**输入：**

- **对应方式：** 面板每行是一笔贷款的一个月。`loan`、`month`、`grade`、`defaulted`，以及可选的 `probability`、`weights`，长度相同。都是 Series 时索引必须相同（`shared_index`），数组按位置对应，与 `vintage_table` 相同。
- **`loan`：** 债务人或债项的编号。缺失值报错；同一笔贷款在同一个月有两行时报错。
- **`month`：** 月份，与 `panel_features` 的 `month` 相同：月度 Period，或代表其所在月的日期时间。表示月末的状态，与 `delinquency_panel` 一致。缺失值和其他类型报错。
- **`grade`：** 当月的等级或池，只用起点那一行（EBA 74）。
  - 缺失值自成一行，排在最后（EBA 75，决定 5）；
  - 分类型按类别顺序，其他类型排序，与 `vintage_table` 的队列相同；
  - 某个起点没有债务人的等级不出行。
- **`defaulted`：** 布尔值或 0 与 1，表示当月是否处于违约。缺失值报错。docstring 说明三点：
  - 应包含月内开始、月内结束的违约和考察期；
  - 应去掉技术性违约（ECB 2.3(d)）和错误记录的违约（EBA 71(a)）；
  - 可以用逾期状态构造，如 `status == "90+"`。
- **`horizon`：** 窗口的月数，正整数，默认 12，即一年违约率。
- **`starts`（决定 2）：** 观察起点的月份。
  - **默认：** 不给时，取面板第一个月起的每个季末月（3、6、9、12 月），只保留窗口在面板最后一个月或之前结束的（EBA 78）。
  - **显式给出：** 可以是 Period、日期时间或字符串。字符串按 `pandas.Period` 读取，取其最后一个月，与 `delinquency_panel` 的 `end` 相同，所以 `"2024Q2"` 表示 6 月。
  - **报错：** 起点早于面板第一个月；窗口超过面板最后一个月；起点重复。窗口不完整会低估违约率，所以不能静默截断。
- **`probability`（决定 4）：** 可选，每行的 PD。起点行的平均值给出 `mean_probability` 列，即 ECB 2.5.3.1 的“期初 PD”，也是 CRE36.78(3) 的简单平均。缺失值只在进入队列的行报错，违约行的 PD 常为空，不检查。
- **`weights`：** 可选，每行非负的频数权重，用起点那一行的。权重和为 0 的组不出行，与 `_counts` 相同。docstring 写明 CRE36.78 不允许按 EAD 加权。

**计算：** 对每个起点 r：

- **队列：** 在 r 月有一行、且该行 `defaulted` 为假的贷款。
- **`event`：** 队列中在 r+1 至 r+`horizon` 月里至少有一个违约月的贷款，只计一次（EBA 77）。
- **`exited`：** 队列中没有违约、且最后一行早于 r+`horizon` 月的贷款，即在窗口内偿清、出售、核销或迁出面板的。它们仍然计入 `total`（EBA 76）。
- **缺月：** 窗口中缺失的月份视为未违约。docstring 提醒：面板应保留期间迁移到其他模型的债务人（EBA 76、ECB 2.3(c)），否则这些违约会漏掉。

**结果：**

- **索引：** DataFrame，二级索引 (`start`, `grade`)，`start` 是月度 Period，按起点、再按等级排序。
- **列：** `total`、`event`、`exited`、`event_rate`，给出 `probability` 时再加 `mean_probability`。没有权重时计数为整数，有权重时为浮点数，与 `calibration_table` 相同。
- **实现：** 按 (loan, month) 排序一次，用每笔贷款违约月数的累计和，以及组合键上的 `searchsorted`，求出每行此后 `horizon` 个月内的违约次数。每个起点 O(n)，不对贷款做 Python 循环。

**报错：**

- 输入为空、不是一维、长度或索引不一致；
- `loan`、`month`、`defaulted` 有缺失值，或同一笔贷款同月有两行；
- `month` 的类型不对；
- `defaulted` 有 0 与 1 以外的值；
- `horizon` 不是正整数；
- `starts` 无效；
- 默认的起点一个也没有，即面板短于 `horizon` 加 1 个月；
- `probability` 或 `weights` 无效。

**指南中的写法（不设函数，决定 9）：**

```python
table = default_rate_table(panel["loan_id"], panel["month"], panel["status"],
                           panel["status"] == "90+")
observed = table["event_rate"].groupby(level="grade").mean()          # EBA 79、81
portfolio = table.groupby(level="start")[["total", "event"]].sum()
portfolio_rate = portfolio["event"] / portfolio["total"]
long_run = portfolio_rate.mean()                                        # EBA 84
benchmark = max(portfolio_rate.iloc[-20:].mean(), long_run)             # EBA 86：最近 5 年为季度起点的最后 20 个
```

### 3.2 `central_tendency_shift`

- **结果（决定 6）：** 返回 log odds 的平移量 Δ，一个浮点数，使加权平均 Σ w_i·expit(logit p_i + Δ) / Σ w_i 恰好等于 `central_tendency`。函数名本身就说明返回的是平移量。
- **应用：** 这是“拟合”，由调用者应用到别处，docstring 和指南给出三种写法：
  - **概率：** `p / (p + (1 - p) * numpy.exp(-shift))`，p 为 0 和 1 时保持不变，不需要 logit；
  - **logistic 模型：** 截距加 Δ，与 `corrected_intercept` 相同；
  - **评分卡：** 分数的 offset 减去 factor·Δ。
- **为什么不直接返回概率：** 校准样本不是应用组合（EBA 88），TTC B 在一个时点上拟合，再用到以后的所有债务人。返回概率时，用户还得从某一行反推 Δ。
- **`probability`：** [0, 1] 内的 PD，例如模型在校准样本上的 PD，或每个等级一个 PD 并以等级的户数作权重。
  - 缺失值报错，与其他拟合函数相同（`conventions.rst`）。
  - 0 和 1 允许，平移后不变。要求至少有一个严格介于 0 与 1 之间、权重为正的 PD，且目标严格介于“PD 为 1 的权重份额”和“1 − PD 为 0 的权重份额”之间，否则报错，错误信息给出可达的范围。
- **`central_tendency`：** 严格介于 0 与 1 之间的有限实数，不能是布尔值。
- **`weights`：** 频数权重，权重和须为正。
- **求解：** 平均 PD 随 Δ 严格上升，所以解唯一。用二分法保护的牛顿法，求到相邻浮点数。
- **性质：** 排序不变（EBA 99）。平均值已经等于目标时，Δ = 0。
- **Notes 写明三点：**
  - 它就是 Tasche（2013）的缩放似然比（式 4.5、4.8a）；
  - 与 `prior_correction` 的关系：后者的 Δ 是固定的 ln ω；
  - 应在 MoC 和下限之前做（EBA 89）。

### 3.3 `quasi_moment_fit`

- **名称（决定 1）：** 改名为 `quasi_moment_fit`，与返回参数的 `platt_fit`、`isotonic_fit`、`vasicek_fit` 同一形式。docstring 的第一句用 “quasi moment matching”，方便检索。
- **模型：** PD(s) = 1 / (1 + exp(−(slope·s + intercept)))，与 `platt_fit` 的曲线相同。
- **结果（决定 8）：** 返回 `(slope, intercept)`，顺序与 `platt_fit` 相同，所以 `platt_predict(score, *quasi_moment_fit(...))` 可以直接用。草稿的顺序是 (intercept, slope)。
- **两个方程：**
  - 加权平均 PD 等于 `central_tendency`；
  - 隐含 AR 等于 `accuracy_ratio`：把每个观测拆成权重为 w·PD 的事件和 w·(1 − PD) 的非事件，所有并列（含自身配对）按一半计。这等于 Tasche（2009）的式 (5.16b) 和（2013）的式 (A.11b)，也等于对拆开的样本计算 `kivi.metrics.gini`。
- **`score`：** 有限的数值分数，必须随风险上升，与 kivi 的其他指标相同；分数越高越好时先取负号。
  - 隐含的 AR 只由曲线本身决定，所以斜率总是正的。方向弄反时不会报错，只会得到上升方向的曲线，docstring 必须讲清楚。
  - 缺失值报错。`accuracy_ratio` 为正时，至少要有两个权重为正的不同分数。
- **`accuracy_ratio`（决定 8）：**
  - 取值范围是 0 ≤ AR < AR_max。AR = 0 时斜率为 0，截距为 logit(中心趋势)。
  - AR_max 是斜率趋于无穷、平均值保持不变时的极限：分数最高的一部分 PD 为 1，分界处的分数取一个中间值，其余为 0。由于并列，它通常小于 1，等级少时明显小于 1。
  - 达到或超过 AR_max 时报错，错误信息给出 AR_max。
- **求解：**
  1. 先把相同的分数合并；
  2. 对给定的斜率，截距由与 `central_tendency_shift` 相同的单调求解得到；
  3. 隐含 AR 随斜率上升（第 2.4 节的数值检验），所以对斜率做有界的一维求解，初值取 Tasche 式 (A.7) 的二元正态近似，直到相邻浮点数。
- **robust logit（决定 7）：** 不设选项，曲线对传入的分数是 logistic 的，即 Tasche 2009 年例 5.4。指南示范 Tasche 2013 年的 robust 版本：对等级先计算 t = Φ⁻¹(F̃(x))，再把 t 传入；F̃ 是无条件分布（第 4.2.2 节）或存活者分布（式 A.9）的中点分布函数，Φ⁻¹ 用标准库的 `statistics.NormalDist().inv_cdf`。测试也用这种写法复现表 5。
- **`weights`：** 频数权重，例如每个等级的户数。整数权重等于重复行。
- **Notes 写明：** 平均 PD 固定时 AR 随斜率单调，这一点是数值检验的结论，不是 Tasche 的定理。

### 3.4 不做的内容

- **长期平均的函数，以及 EBA 85 的调整和 MoC：** 属于机构的判断，指南给写法（决定 9）。
- **缩放 PD（Tasche 式 4.4）：** 理论上有问题，结果可能超过 1，指南提一句。
- **不变违约分布（Tasche 式 4.2）：** 和由评级分布预测 p₁（第 4.4 节）。Tasche 的表 8 显示后者在 S&P 数据上的预测偏差很大。
- **重叠窗口的偏差修正（EBA 80(c)）：** 由机构分析，指南说明重叠窗口使中间年份的权重更高。

## 4. 测试与参考

| 内容 | 对照 |
| --- | --- |
| `default_rate_table` | 手算小面板：退出、违约后治愈再违约只计一次、起点已违约的不计入、缺失评级自成一行、期间迁移不改变起点等级。在含缺月、退出和多次违约的随机面板上，与逐个起点、逐笔贷款循环的暴力实现完全一致。单个起点与 `calibration_table` 的计数一致。窗口内没有退出时，与 `lifetime.survival_table` 的 1 − S(12) 一致。整数权重等于重复贷款。`make_loan_panel` 到 `default_rate_table` 的完整流程。各项报错 |
| `central_tendency_shift` | 加权平均等于目标，相对误差 1e-12；排序不变；已校准时为 0；PD 为 0 和 1 时不变，可达范围的边界报错；与 SciPy `brentq` 的解一致；以 Tasche（2013）表 5 印出的 PD 为输入，复现表 7 的“缩放似然比”列，绝对误差在 0.0004 个百分点以内（实测最大 0.0003；输入只印到 3 位小数）；与 `prior_correction` 的等价情形；整数权重等于重复行 |
| `quasi_moment_fit` | 以 Tasche（2013）表 2 为输入：`gini` 复现表 4 的 82.7%；用 robust logit 的变换复现表 5，17 格中 15 格印出的位数完全相同，另两格（BB、CCC-C）差末位，在测试中列出，与 1.5 处理 Pluto–Tasche 表格的做法相同；复现表 7 的“不变 AR”列，绝对误差 0.0005 个百分点。拟合曲线的平均 PD 和 `kivi.metrics.gini` 的隐含 AR 等于目标，相对误差 1e-12。与 SciPy `optimize.root` 加 scikit-learn `roc_auc_score(sample_weight=)` 的解一致，相对误差 1e-8。分数的仿射变换不改变 PD。AR = 0 时斜率为 0。AR_max 处报错。整数权重等于重复行 |

- **数据：** Tasche 的 S&P 数据（表 2 的计数、表 5 和表 7 的值）写死在测试里。复现所用的变换和目标也在测试里计算，不依赖 SciPy。
- **交叉核对：** 用 SciPy 和 scikit-learn 的核对先放在 `tests/test_calibration.py`，用 `pytest.importorskip`，由集成 PR 移入 `tests/test_reference.py`。

## 5. 会话与流程

| PR | 文件 |
| --- | --- |
| 本稿 | `docs/design/kivi-1.8-default-rates.md`（新），`docs/design/kivi-1.8-roadmap.md` 第 3.3 节改为指向本稿 |
| 实现，审定后开工 | `src/kivi/vintage.py`、`src/kivi/calibration.py`、`tests/test_vintage.py`、`tests/test_calibration.py`、`tests/test_typing.py`、`docs/user_guide/vintage.rst`（新的“一年违约率”一节）、`docs/user_guide/calibration.rst`（新的“对准中心趋势”一节）、`docs/api/vintage.rst`、`docs/api/calibration.rst`、发布说明片段，以及 `kivi.po` |

- **`tests/test_typing.py`：** 按推荐方案，两个校准函数都返回浮点数或元组，结果的类型不随输入变化，不需要 `typing.overload`。决定 6 若选“返回概率”，就要按 `prior_correction` 的样子加重载声明和检查。
- **集成 PR 需要补的内容**（实现 PR 的描述中会再列一次）：
  - `compatibility.rst`：`default_rate_table` 的索引 (`start`, `grade`) 和列，`quasi_moment_fit` 的返回顺序；
  - `conventions.rst`：返回参数的元组里加上 `quasi_moment_fit`，缺失值一节加上新函数；
  - `glossary.rst`：一年违约率、队列法、观察起点、观察平均违约率、长期平均违约率、中心趋势、校准分段、校准样本、quasi moment matching、AR\*、缩放似然比、TTC B/C；
  - README 的 API 表（中英两版）；
  - 交叉核对移入 `test_reference.py`；
  - 一张图：各起点的违约率与长期平均，放进 `_plotting.py` 或 `plotting-tables` 会话之后的图表 PR。

流程：

1. 本稿审定：等你对第 6 节的决定。
2. 实现 PR：从最新的 master 分出，完成后开 ready PR，列出公共 API 和集成 PR 的注意事项。不合并。
3. 集成 PR 和发布 1.8.0 由协调会话按你的指示进行。

## 6. 决定

| # | 问题 | 推荐 | 其他选项 |
| --- | --- | --- | --- |
| 1 | 函数名（路线图第 7 节第 10 项） | `default_rate_table`（与 `calibration_table`、`vintage_table` 同形）、`central_tendency_shift`（返回的就是平移量）、`quasi_moment_fit`（与 `platt_fit`、`isotonic_fit`、`vasicek_fit` 同形） | 草稿的 `quasi_moment_matching`：方法名，但它是唯一不带 `fit` 而返回参数的拟合函数 |
| 2 | 观察起点 | 关键字 `starts`、索引层 `start`；默认取每个季末，只保留窗口完整的（EBA 78）；显式给出的起点窗口不完整时报错 | 草稿的 `reference`：与 PSI 的参照样本 `reference`、`reference_period` 冲突；`cohort`：与本模块中表示放款月份的 `cohort` 冲突；默认每月一个起点 |
| 3 | 列名与等级参数 | `total`、`event`、`event_rate`，与 `calibration_table`、`survival_table` 一致，可以直接交给 `jeffreys_test(t["event"], t["total"], ...)`；参数和索引层都叫 `grade`（EBA 74 的“等级或池”） | 草稿的 `total`、`default`、`default_rate`；参数 `groups`、索引层 `bin`，与 `calibration_table` 相同 |
| 4 | 附加列 | `exited`：窗口内未违约就离开面板的户数，供 EBA 76、80(a) 的偏差分析；`probability` 给出时加 `mean_probability`，供 ECB 2.5.3.1 的逐等级检验 | 只有三列，其余由调用者自己算 |
| 5 | 缺失评级 | 自成一行，排在最后，与 `_counts` 相同：EBA 75 要求单独计算，ECB 的口径去掉这一行即可 | 报错，由调用者先标记 |
| 6 | `central_tendency_shift` 的结果 | 平移量 Δ（浮点数）。校准样本上拟合、应用于整个模型（TTC B），截距、评分卡 offset 和概率的应用写法都在文档中 | 草稿：返回形状同输入的概率，需要 `typing.overload`，应用到新数据时要反推 Δ；或者返回 Δ，另加一个应用函数，日后再加也不破坏兼容 |
| 7 | QMM 的曲线 | 对传入的分数做 logistic（Tasche 2009 例 5.4）；robust logit 由调用者先变换分数，指南示范，测试用它复现 Tasche 2013 表 5 | 加 `robust=True`，函数内部按校准样本的中点分布做 Φ⁻¹ 变换：新分数要用同一个变换，`platt_predict` 无法直接应用 |
| 8 | QMM 的参数和结果 | 目标参数叫 `accuracy_ratio`（Tasche 和 IRB 验证文献的用语，`metrics.gini` 的 docstring 已注明两者相同），取值 0 ≤ AR < AR_max；返回 `(slope, intercept)`，与 `platt_fit` 同序 | 参数叫 `gini`（kivi 的列名）或 `auc`（ECB 报告的用语）；草稿的返回顺序 (intercept, slope) |
| 9 | 长期平均 | 不设函数（重新确认路线图的决定）：观察平均是一行 `groupby().mean()`，EBA 85 的调整是机构判断，指南给出 EBA 79 的两个层面、EBA 81 零售加权平均和 EBA 86 基准的写法 | `long_run_default_rate(table, *, level, weights)` |
