# kivi 1.10 设计：转移矩阵的单因子模型

2026-11-26 起草 · 待审定

## 1. 目标与范围

`docs/design/kivi-1.10-roadmap.md` 第 5.2 节给 `matrix-scenarios` 会话排了两个暂名的函数，放在 `kivi.lifetime`：

- **`shift_matrix`：** 按系统因子平移整个转移矩阵；
- **`matrix_factor`：** 从某一期观察到的计数矩阵估计该期的因子。

2026-11-26 用户采用路线图的建议（“Yes, both sessions”），其中决定 3 是：采用 Belkin、Suchower 和 Forest（1998，下称 BSF）的模型，先核对原文，再审定 API，与 1.8 的 `default-rates` 相同。名称按决定 9 暂定，在 PR 审阅时确认。

本稿做四件事：

1. 记下核对结果和出处，说明哪些没有核对到（第 2 节）；
2. 据此给出 API，并按原文的做法增加一个估计相关系数的函数（第 3 节）；
3. 给出测试依据和指南的改写方案（第 4、5 节）；
4. 列出文件、流程和待你决定的事项（第 6、7 节）。

第 3 节起，第一个函数按决定 1 的推荐称为 `conditional_matrix`，与 `capital.conditional_pd` 对应；如果保留 `shift_matrix`，只改名称。

1.10 不需要新的运行时依赖：三个函数只用 NumPy、pandas 和 `_stats` 的正态分布函数与分位数。

## 2. 原文核对结果

2026-11-26 核对。

### 2.0 文献

**没有取得原文的：**

- **BSF 原文：** B. Belkin, S. Suchower, L. R. Forest, “A one-parameter representation of credit risk and transition matrices”, *CreditMetrics Monitor*, 1998 年第三季度。
  - MSCI 网站的这一期页面在目录中列出了这篇文章（第 5 篇），但没有 PDF 可下载。
  - 网上唯一的副本在 Scribd 上，需要付费登录，没有读取。
  - Witzany 的参考文献引作 *CreditMetrics Monitor* 1(3), 46–56；Shao 和 Đurović（2019，*Journal of Central Banking Theory and Practice* 8(1)）引作 KPMG Peat Marwick 的工作论文（1998）。页码未核对。
  - **因此本稿没有逐字核对 BSF 原文。** 下文凡说“BSF 的做法”，依据的都是后来文献的转述，逐条注明出处；几份转述在要点上一致，但不能代替原文。
- **Kim（1999）：** MSCI 网站上 Jongwoo Kim 的 “A Way to Condition Transition Matrix on Wind”，1999 年 5 月 1 日。页面只有摘要，下载按钮没有链接，没有全文；它是否就是任务说明所说的 *Risk* 文章，也未核对。摘要说：先用最相关的宏观和金融序列构造一个信用周期指数，再按预测的指数调整转移矩阵，用于压力测试。公式未核对。
- **其他：** Bangia 等（2002，PIER 工作论文 01-004）和 Jones（2006）连接失败；S&P Global Market Intelligence 的 “Conditioning credit risk transitions on macro-economic scenarios”（2017 年 12 月）拒绝访问（403）。

**核对了的文献：**

| 简称 | 文献 | 用到的部分 |
| --- | --- | --- |
| CMTD | G. M. Gupton, C. C. Finger, M. Bhatia, *CreditMetrics – Technical Document*，J.P. Morgan 1997；RiskMetrics Group 2007 年重印本，取自 MSCI 的 PDF | 第 8.4 节第 87–88 页：式 [8.2]、[8.3]，表 8.4、8.5 |
| Witzany | J. Witzany, “Stressing of migration matrices for IFRS 9 and ICAAP calculations”，FFA Working Paper 1/2020，布拉格经济大学（期刊版载于 *Journal of Credit Risk* 18 (2022) 49–76，未核对） | 第 2 节式 (1)–(4)、(8)–(10) 和步骤 1–5（第 3–6 页），表 1（第 7 页），表 2–4（第 9 页），IFRS 9 的例子（第 12 页） |
| Engelmann | B. Engelmann, “Spurious default probability projections in credit risk stress testing models”，arXiv:2401.08892v1（2024 年 1 月 16 日） | 第 2 节式 (1)–(8)（第 2–4 页），第 3 节式 (9)、(10) 和定理 1（第 5–6 页），第 4 节（第 7–10 页） |
| Shao | J. Shao, S. Li, Y. Li, “Estimation and prediction of credit risk based on rating transition systems”，arXiv:1607.00448v2（2018 年 3 月 27 日） | 第 1 节式 (1.1)–(1.7)（第 2–4 页）转述 BSF；第 2 节（第 5–7 页）转述 BSF 选取 ρ 的方法并提出批评 |
| CERM | J. Garnier, J.-B. Gaudemet, A. Gruz, “The Climate Extended Risk Model (CERM)”，arXiv:2103.03275v2（2022 年 4 月 10 日） | 第 4.1 节式 (12)、(13) 和脚注 6（第 10 页），第 5.1 节式 (27)、(28)（第 13–14 页），都引用 BSF |
| Leegstra | M. Leegstra, “The influence of climate-related and environmental risks on bank loan credit migration”，鹿特丹伊拉斯谟大学硕士论文，2023 年 7 月 10 日 | 第 3.1.1–3.1.2 节式 (3.1)–(3.4)（第 15–16 页），第 3.5 节式 (3.20)–(3.22)（第 20–21 页），第 5.1.1 节（第 32 页）。这是硕士论文，只用作 BSF 估计步骤的第二份转述 |

Witzany 和 Engelmann 是使用并检验该方法的文献：前者用于 IFRS 9 和 ICAAP，后者用于多年期压力测试。Shao 和 Leegstra 详细转述了 BSF 的估计步骤。

### 2.1 阈值：从每一行的累计概率得到

| 出处 | 内容 |
| --- | --- |
| CMTD 式 [8.2]、[8.3]（第 87–88 页） | 资产收益低于 Z_Def 时违约，介于 Z_Def 与 Z_CCC 之间时降到 CCC，依此类推；先由违约概率解出 Z_Def = Φ⁻¹(p_D)·σ，再由 CCC、B…… 的概率逐个向上解出其余阈值；最好的等级 AAA 没有上阈值 |
| Witzany 式 (8)、(9)（第 5 页） | 阈值 b_{u,v} 递减，PD_{uv+} = PD_{uv} + … + PD_{uK} 是“到 v 或更差”的累计概率，b_{u,v} = Φ⁻¹(PD_{uv+}) |
| Engelmann 式 (5)（第 3 页）、CERM 式 (13)（第 10 页）、Leegstra 式 (3.2)、(3.3)（第 15 页） | 同样从违约状态向上累计：z_{ij} = Φ⁻¹(Σ_{j′≥j} M_{ij′}) |
| Shao 式 (1.3)（第 3 页） | 从等级 1 向下累计：x̂ʲ = Φ⁻¹(Σ_{k≤j} P̄_ik)，违约是第 K 个等级（第 6 页）。与上面的写法只差因子的符号 |

**状态的顺序：** 状态按从最好到最差排列，违约最差。阈值由“该状态或更差”的累计概率给出，信用指数越低，状态越差。

**数值核对：** CMTD 表 8.4 的 BB 行（AAA 到违约依次为 0.03%、0.14%、0.67%、7.73%、80.53%、8.84%、1.00%、1.06%）按上式得到的 7 个阈值，保留 2 位小数后与表 8.5 完全相同：

| 阈值 | Z_AA | Z_A | Z_BBB | Z_BB | Z_B | Z_CCC | Z_Def |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 表 8.5（σ 的倍数） | 3.43 | 2.93 | 2.39 | 1.37 | −1.23 | −2.04 | −2.30 |

表 8.5 的标题写的是 “BBB rated obligor”，但正文和数值都是 BB 的例子。

**对实现的影响：** 两端的累计概率都要算，每个阈值取较小的一侧求分位数：“该状态或更差”不超过 0.5 时用它，否则用“更好的状态”之和取负号。条件概率的差也取较小的尾部，这样极小的格子也保持相对精度。原型中，1e-11 的格子在 ρ = 0 时的相对误差为 7e-15。两端的累计和要分别从各自一端起算，不能用 1 减去另一端：后者在原型里给出 8e-8 的相对误差。

### 2.2 给定因子的条件概率

**模型：** 信用指数 X = √ρ·Z + √(1 − ρ)·Y，Z 和 Y 是独立的标准正态变量。出处：Witzany 式 (1)，Engelmann 式 (1)，Shao 式 (1.1)，Leegstra 式 (3.4)；Leegstra 写作 α = √ρ。

**条件概率：**

- “到 j 或更差”：P_{ij+}(z) = Φ((b_{ij} − √ρ·z) / √(1 − ρ))（Witzany 式 (10)，CERM 式 (27)）；
- 每个格子：相邻两个累计概率之差（Witzany 第 6 页，Engelmann 式 (5)，CERM 式 (28)）。

**因子的符号：**

- 正的 z 表示经济景气（Engelmann 第 2 页）；
- Witzany 以 Z = Φ⁻¹(0.01) 表示 1% 概率的压力情景（第 9 页）；
- Leegstra 写明正的 ΔZ_t 表示违约较少、信用恶化较少的好年份（第 21 页）；
- Shao 从最好的等级累计，正的 z 反而使状态变差。

kivi 采用前三者的约定，与 `capital.conditional_pd` 相同：正的因子表示经济好于平均。违约列就是 `conditional_pd(p_iD, correlation=ρ, factor=z)`，Witzany 的式 (2) 与式 (10) 在违约列上相同。

**因子为 0 不等于长期矩阵：**

- ρ > 0 时，z = 0 给出中位经济下的矩阵。长期矩阵是条件矩阵对 Z ~ N(0, 1) 的平均：Shao 第 3 页指出，各期条件概率的平均几乎必然收敛到 Φ(x_j) − Φ(x_{j−1})。这与 `conditional_pd` 的 docstring 一致：其 Notes 写明对因子取平均才是长期 PD，例子中因子为 0 的 PD 低于长期 PD。
- **数值：** 以 Witzany 表 2 为例，ρ = 0.08 时，z = 0 的矩阵与表 2 最多相差 1.95 个百分点；对 z 做 80 个节点的 Gauss–Hermite 积分，结果与表 2 相差不超过 7e-16。
- **对任务说明的修正：** 任务说明的测试计划写了“Z = 0 时返回原矩阵”，这只在 ρ = 0 时成立。第 4 节改为两项：ρ = 0 时返回原矩阵；对因子取平均时返回原矩阵（决定 2）。
- **对指南的影响：** IFRS 9 情景超出预测期以后，应当回到长期矩阵本身，而不是因子为 0 的矩阵。

**参考值：Witzany 表 2 和表 3（第 9 页）。**

- **输入：**
  - 表 2 是 Moody's 1920–2011 年的 TTC 一年矩阵；
  - 表 3 是以 ρ = 0.08、Z = Φ⁻¹(0.01) 平移后的矩阵；
  - 两表都以百分数印到 3 位小数。
  - 表 2 的行和与 100% 相差不超过 0.002 个百分点，所以先把各行除以行和。
- **结果：** 8 个非违约行共 72 格，原型与表 3 最大相差 0.009 个百分点。
- **输入舍入的范围：** 给表 2 的每格加上 ±0.0005 个百分点的均匀扰动，做 200 次。72 格中 66 格落在扰动结果加减表 3 舍入的范围内。其余 6 格是：

| 格子 | 表 2 | 表 3 | 原型 |
| --- | --- | --- | --- |
| Aaa → 违约 | 0.000 | 0.009 | 0.0000 |
| A → 违约 | 0.102 | 0.578 | 0.5709 |
| Baa → 违约 | 0.303 | 1.474 | 1.4802 |
| Ba → 违约 | 1.433 | 5.530 | 5.5324 |
| B → 违约 | 4.187 | 13.196 | 13.1995 |
| B → B | 81.429 | 70.683 | 70.6806 |

- **差异出在 Witzany 的输入：**
  - `kivi.capital.conditional_pd` 对表 2 的违约列给出与原型相同的值，而 Witzany 正文也写 A 的压力 PD 为 0.578%（第 9 页）；
  - 由表 3 的违约列反推的因子在 −2.341（A）与 −2.321（Baa）之间，而 Φ⁻¹(0.01) = −2.326；
  - 偏差方向不一，改用表 1 的 ρ = 0.07969 也解释不了。最可能的是，他用的是未经四舍五入、与印出的表 2 略有不同的输入。
  - Aaa 行表 2 印为 0.000 的格子实际上是一个很小的正数。
- **多年矩阵：** 表 4（三年）与表 3 的三次方最多相差 0.0015 个百分点，说明 Witzany 的多年矩阵就是逐年相乘。原型三次方与表 4 最多相差 0.021 个百分点，来自上面的违约列差异。
- **IFRS 9 的例子（第 12 页）：**
  - 情景 Z = −1、−2.15、0.15，权重 50%、25%、25%；
  - Witzany 给出 Baa 的三年 PD 为 3.03%（图 7 是 Baa）；
  - 原型在 ρ = 0.08 时为 3.04%，ρ = 0.07969 时为 3.035%。

### 2.3 违约保持吸收

- **Engelmann 式 (4)（第 3 页）：** 违约行为 (0, …, 0, 1)，写明“不可能治愈”。
- **CERM 第 4.1 节（第 10 页）：** (M)_KK = 1。脚注 6 说，违约不是吸收态时（如主权）需要另外的处理。
- **Witzany 第 5 页：** 第 K 个状态是违约这一吸收态，表 2 的违约行为 100%。
- **Leegstra 第 20 页：** 在平均矩阵上加入违约行，从违约到违约为 100%。

**kivi：** 结果的违约行换成单位行，与 `transition_pd` 相同。输入的违约行不使用，可以是 NaN：`transition_matrix(..., normalize=True)` 在期初没有违约贷款时会给出 NaN。

### 2.4 每一期因子的估计

**准则：** Shao 式 (1.5)（第 3 页）和 Leegstra 式 (3.20)、(3.21)（第 20 页）的转述相同：

```
min over z:  Σ_i Σ_j  n_i · (p̃_ij − P_ij(z))² / (P_ij(z) · (1 − P_ij(z)))
```

- **i 和 j：** i 取除违约外的期初状态，j 取全部期末状态。
- **观察值：** p̃_ij 是该期观察到的队列频率 N_ij / N_i（Shao 式 (2.1)，第 6 页）。
- **权重：** n_i 是该期期初在状态 i 的债务人数（Shao）或从 i 出发的迁移数（Leegstra），两者相同。
- **加权方式：** 每个格子的平方差乘以户数，再除以拟合概率处的二项方差，即 Pearson 卡方，或称最小卡方估计。分母用的是随 z 变化的拟合概率 P_ij(z)，不是长期概率。

**长期矩阵：**

- 阈值来自历史平均矩阵。Shao 式 (1.3) 用各期矩阵的简单平均；Leegstra 第 20 页用平滑后的历史平均矩阵。
- 用户也可以把各期计数合并后归一化，等于按户数加权的平均。kivi 不规定用哪一种，由调用者传入 `matrix`，docstring 说明两种做法。

**两个状态的特例：**

- 只有“正常 → 违约”一行时，准则只剩一项二项式，当 P(z) 等于观察到的违约率时取最小值 0。
- 这时的因子就是 `capital.systematic_factor`，即 Witzany 式 (3)（第 4 页）。测试用这一点衔接已有函数。

**唯一性：**

- 文献没有讨论准则对 z 是否单峰。
- 原型做了 60 组随机多项分布计数：ρ 取 0.01–0.4，每行 5–400 户，在 [−6, 6] 上以 0.01 为步长取网格，每组只有一个局部最小值，求解结果与网格最小点相差不超过半个步长。
- 与 1.8 相同，docstring 写明这是数值检验的结论，不是定理。

**文献没有讨论、需要 kivi 规定的情形：**

- **长期概率为 0 的格子：**
  - 无论 z 取何值，条件概率都是 0，该项是 0/0。
  - Leegstra 第 20 页先把平均矩阵平滑到“没有 0 的转移概率”，并归于 BSF 和 Bolder（2022）；这一归属未能在原文核对。
  - kivi 不做平滑：这样的格子不提供信息，不计入准则。该期在这样的格子里有正的计数时，观察结果在长期矩阵下不可能发生，报错。
  - 长期概率为 1 的格子同理：整行只有一个状态，不提供信息。
- **该期没有户数的行：** 权重 n_i = 0，自然不计入。除违约行外至少要有一行有户数。
- **因子不是有限值：**
  - 每一行的观察都落在该行可达的最好状态时，准则随 z → +∞ 降到 0，没有有限的最小值；最差状态时同理，z → −∞。
  - 这在户数很少的期最容易出现。kivi 报错，不返回 ±∞。
- **数值：** z 很极端时，概率很小的格子的拟合概率会下溢为 0。观察为 0 的格子取其极限 0；原型最初在这里得到 0/0。

### 2.5 相关系数 ρ 的选取

**BSF 的做法：**

- **转述：** Shao 第 6 页说，BSF 建议先假设一个 ρ，按式 (1.5) 得到各期的 Z_t；对许多 ρ 重复这一步，用数值搜索找出使 Z_t 序列方差为 1 的 ρ。Leegstra 第 21 页的转述相同，并说明对不同的 α 计算 Z_t 的均值和方差。
- **均值：** 不强制为 0。Leegstra 的结果是 α = 0.186，ΔZ_t ~ N(−0.169, 1.000)（第 32 页）；按他的式 (3.4)，ρ = α² ≈ 0.035。
- **方差的口径：** 两份转述都没有说方差除以 T 还是 T − 1。`capital.vasicek_fit` 用 `ddof=1`，本稿建议相同（决定 4）。

**文献中的其他做法：**

- **Witzany 步骤 1（第 6 页）：**
  - 从违约率序列估计 ρ 和各期的 z_t，用 Yang（2013）的极大似然（式 (4)）；
  - 他说这是银行在 IFRS 9 和 ICAAP 建模中常用的做法（第 5–6 页）；
  - 表 1 显示，按全部等级、投资级、投机级和各等级分别估计的 ρ 相差很大（Probit 从 6.6% 到 30.7%）；
  - 他指出单个等级的违约率序列噪声比合并的序列大得多（第 8 页）。
- **Engelmann 式 (7)（第 4 页）：** 从信用指数（如违约率序列）估计，用 Carlehed 和 Petrov（2012）的矩估计或 Duellmann、Küll 和 Kunisch（2010）的方法。
- **kivi 已有的工具：** `capital.vasicek_fit`（矩估计，`ddof=1`）和 `systematic_factor` 正是这条路线。

**批评（Shao 第 4–6 页）：**

- Z 服从标准正态的假设难以检验；
- ρ 难以估计，用其他方法（如巴塞尔公式）得到的 ρ 不能保证该假设；
- 每期只有 K(K − 1) 个格子可用。

**原型的数值检验：**

- **设定：** 以 Witzany 表 2 为真实矩阵做 4 组模拟，真实 ρ 取 0.03–0.25，12–20 期，每期每行几十到几千户，长期矩阵取各期计数的合并。
- **单调性：** Z_t 的样本方差都随 ρ 严格下降，在 ρ → 0 时趋于无穷，所以可以对 ρ 做二分。
- **精度：** 估计受期数限制。真实 ρ = 0.10、15 期时，一次模拟得到 0.036，原因是那 15 个真实因子的样本方差远小于 1；真实 ρ = 0.25、12 期时得到 0.258。每期户数少时，二项噪声使 Z_t 的方差变大，ρ 的估计偏高，与 `vasicek_fit` 的说明相同。

### 2.6 多年路径

- **Witzany 步骤 5（第 6 页）：** 对每个未来期 t，用式 (10) 按 z_t 调整 TTC 矩阵，再把各期矩阵相乘，得到更长期的迁移概率。第 9 页说，第二年、第三年可以用不同的 Z；表 4 是表 3 的三次方。
- **Engelmann 式 (9)（第 5 页）：** 每期用 T(z_t) 推进组合。第 3–4 节指出两个问题：
  - 用固定矩阵反复推进，组合会收敛到一个只由参数决定、与当前组合无关的 TTC 组合（定理 1，第 6 页）；
  - 当前组合与矩阵不一致时，即使不加压力，推进的头几年也可能出现虚假的衰退或景气（图 1–3）。
  - 他建议先不加压力推进一次，检查平均 PD（第 9–10 页）。
- **CERM：** 每期一个矩阵 M_{g,t}。

**与 `transition_pd` 的衔接：**

- 每期调用一次 `conditional_matrix`，把结果按时间顺序相乘，乘积的违约列就是到该期为止的累计 PD；
- 违约行是单位行，所以乘积同样把治愈的贷款计为违约，与 `transition_pd` 的口径一致；
- `transition_pd(乘积, horizons=1)` 给出路径终点的累计 PD；逐期的累计 PD 用一个循环；
- 预测期以后用 `matrix` 本身（第 2.2 节）。

路线图写的“每年用不同的因子时逐年相乘”与此相同，本稿不改 `transition_pd`（决定 9）。

### 2.7 零售逾期矩阵

**文献的对象：** 文献都以评级为对象。逾期桶同样有顺序：`current`、`1-29`、`30-59`、`60-89`、`90+`。一个月最多后退一个桶，所以月度矩阵有许多结构性的 0；按第 2.1 节的阈值，这些格子的阈值相等，任何因子下都保持 0。

**原型在 `make_loan_panel` 上的结果：**

- **估计方法：** 按月把相邻两个月的状态配对，每月一个计数矩阵；长期矩阵取合并计数，即指南中的 `monthly`；按单位方差估计 ρ，再估计每月的因子。
- **默认面板（24 个月，23 期）：** ρ = 0.0086。因子在 2024 年多为负，2025 年转正；2024 年和 2025 年的平均分别为 −0.78 和 0.24。
- **`n_months=60`：** ρ = 0.0118，2024–2028 年的平均依次为 −1.16、−0.70、0.48、0.17、0.05。
- **原因：** 面板没有经济周期，见 `make_loan_panel` 的 Notes。因子反映的是账簿的账龄：2024 年的贷款大多处在第 6 个月前后的账龄高峰，以后账簿变老。
- **对指南的意义：** 因子吸收所有使各行一起变好或变差的变化，包括组合结构的变化。所以长期矩阵和各期矩阵应当来自可比的组合，必要时先按账龄或产品分段。

**因子和 ρ 属于矩阵的期间：**

- 用月度矩阵估计的 ρ 和因子都是月度的。指南一直用月度矩阵，情景的因子也就按月给出。
- 以 ρ = 0.0086 计，因子 1、0、−1.5 在前 12 个月保持不变、之后回到长期矩阵时，`current` 贷款的 12 个月 PD 分别为 2.51%、3.86% 和 6.94%，长期矩阵为 3.93%。
- 现行指南假设 ρ = 0.1；在同样的设定下，三个值为 0.56%、3.07% 和 19.77%。

### 2.8 kivi 已有的部分

| 函数 | 与本稿的关系 |
| --- | --- |
| `vintage.transition_matrix` | 计数（`matrix_factor` 和 `matrix_correlation` 的 `counts`）和归一化的矩阵（`matrix`）。`states` 应按从最好到最差排列 |
| `lifetime.transition_pd` | 矩阵的幂和吸收的违约行。`conditional_matrix` 沿用它对 `matrix` 和 `default_state` 的检查和报错；多年路径见第 2.6 节 |
| `capital.conditional_pd` | `conditional_matrix` 的违约列，测试逐格比较 |
| `capital.systematic_factor` | 两个状态时的 `matrix_factor`，测试逐个比较 |
| `capital.vasicek_fit` | 从违约率序列估计 ρ，即 Witzany 的步骤 1，是 `matrix_correlation` 之外的另一条路线；指南并列两者。方差同样用 `ddof=1` |
| `_stats.normal_cdf`、`normal_quantile` | 正态分布函数和分位数。`normal_cdf` 的下尾保持相对精度，上尾用 `normal_cdf(-x)` |

## 3. API

```python
# kivi.lifetime
conditional_matrix(matrix, *, correlation, factor, default_state=None)
# -> DataFrame：给定因子的转移矩阵，标签与 matrix 相同，各行之和为 1，违约行为单位行
matrix_factor(counts, matrix, *, correlation, default_state=None)
# -> float：一期计数相对于长期矩阵 matrix 的因子
matrix_correlation(counts, matrix, *, default_state=None)
# -> float：使各期因子的样本方差为 1 的相关系数
```

三个函数的结果类型都不随输入变化，不需要 `typing.overload`，也不改 `tests/test_typing.py`。

### 3.1 `conditional_matrix`

**`matrix`：** 一期的转移概率矩阵，与 `transition_pd` 的要求相同。

- 二维、方阵、实数；DataFrame 的索引必须等于列且没有重复，数组的状态为位置 0、1、……；
- 违约行以外的值在 [0, 1] 内，没有缺失值，各行之和在 1e-9 以内等于 1。发布的矩阵经过四舍五入，先除以行和，与 `transition_pd` 的说明相同；
- **新增的要求：** 状态的顺序就是信用的顺序，从最好到最差。这一点无法检查，docstring 写明；`transition_matrix` 的结果按 `states` 排列，排序的标签会把 `current` 排在 `90+` 之后，与它的说明相同。

**`correlation`：** [0, 1) 内的有限实数，不能是布尔值，范围与 `conditional_pd` 相同，但只接受标量（决定 8）。ρ = 0 时，任何因子都返回原矩阵，违约行除外。

**`factor`：** 有限实数，不能是布尔值。正值表示经济好于平均，违约和降级减少、升级增加。只接受标量：多年路径是逐期调用。

**`default_state`：**

- 含义与 `transition_pd` 相同，默认为最后一个状态；
- 它的行换成单位行；
- 计算阈值时它是最差的状态：其余状态按 `matrix` 的顺序从最好到最差排列，违约排在它们之后，与它在 `matrix` 中的位置无关（决定 7）；
- 结果保持 `matrix` 原来的顺序。

**计算：** 对违约行以外的每一行 i：

1. **阈值：** 第 j 个状态的“该状态或更差”的累计概率 C_ij，b_ij = Φ⁻¹(C_ij)。累计和从两端分别起算，取较小的尾部（第 2.1 节）。
2. **条件累计概率：** Φ((b_ij − √ρ·z) / √(1 − ρ))。
3. **格子：** 相邻两个条件累计概率之差，取较小的尾部计算。

长期概率为 0 的格子，两个阈值相等，结果恰好为 0；为 1 的格子恰好为 1。

**结果：**

- DataFrame，索引和列与 `matrix` 相同，含名称（如 `before`、`after`）；数组输入时两者都是 `RangeIndex`；
- 值为浮点数，各行之和在舍入误差内等于 1；
- 违约行是单位行；
- 可以直接交给 `transition_pd`，或与其他期的结果相乘。

**报错：** `matrix` 和 `default_state` 的情形与 `transition_pd` 相同；`correlation` 不是实数、是布尔值、缺失或不在 [0, 1) 内；`factor` 不是实数、是布尔值、缺失或不是有限值。

**Notes 写明：**

- 模型、公式和因子的符号；
- 因子为 0 是中位经济，对标准正态的因子取平均才是 `matrix`（第 2.2 节）；
- 违约列等于 `conditional_pd`；
- 因子和相关系数属于矩阵的期间；
- 出处：CMTD 第 8.4 节、BSF（1998）、Witzany 式 (10)。

**例子：** 沿用 `transition_pd` docstring 的三状态矩阵，给出 ρ = 0.1、因子 −1 的结果，再交给 `transition_pd`。

### 3.2 `matrix_factor`

**`counts`：** 一期的迁移计数。

- 例如 `transition_matrix(before, after, states=...)`，不归一化；有 `weights` 时是权重之和，即频数权重；
- DataFrame 的索引和列必须与 `matrix` 的完全相同；数组必须形状相同；
- 值非负、有限、没有缺失；
- 违约行不使用。

**`matrix`：** 长期矩阵，要求与 `conditional_matrix` 相同。

**`correlation`：** (0, 1) 内的有限实数。ρ = 0 时矩阵与因子无关，报错，与 `systematic_factor` 相同。

**结果：** 浮点数，即第 2.4 节准则的最小点。

- 计数乘以同一个正数不改变结果，整数权重等于重复的行；
- 期初没有户数的行不计入；
- 长期概率为 0 或 1 的格子不计入。

**求解：**

- 准则对 z 连续可导。先在一个足够宽的区间上取粗网格，找到最小的网格点，再在它两侧的两个网格点之间，对准则的导数（有解析式）做二分，求到相邻浮点数。直接最小化准则只能把 z 求到约 1e-8，因为最小点附近准则很平；
- 区间的边界由阈值和 ρ 决定，使所有格子的条件概率在边界处都已饱和；
- 粗网格是对多峰情形的保护，第 2.4 节的检验中没有出现多峰。

**报错：**

- `counts` 不是二维、形状或标签与 `matrix` 不同，有负值、缺失值或非有限值；
- 除违约行外没有任何计数；
- 在长期概率为 0 的格子里有正的计数，信息给出状态对；
- 没有有限的最小点（第 2.4 节），信息说明观察都落在最好或最差的可达状态；
- `matrix`、`correlation` 和 `default_state` 的情形同 `conditional_matrix`。

**Notes 写明：**

- 准则和权重，出处为 BSF，经 Shao 式 (1.5) 和 Leegstra 式 (3.20) 转述；
- 两个状态时等于 `systematic_factor`；
- 单峰性是数值检验的结论；
- 因子吸收组合结构的变化（第 2.7 节）。

### 3.3 `matrix_correlation`

**`counts`：** 各期的计数矩阵。

- 可以是列表、元组或其他可迭代对象，如 `dict.values()`，至少两期；
- 每一期的要求同 `matrix_factor`；
- 单个 DataFrame 或二维数组报错：对它们迭代会得到列或行，不是各期（决定 5）。

**`matrix`、`default_state`：** 同 `matrix_factor`。

**结果：** (0, 1) 内的浮点数 ρ，使各期的 `matrix_factor(counts_t, matrix, correlation=ρ)` 的样本方差（`ddof=1`）等于 1。均值不强制为 0。

**求解：**

- 样本方差在 ρ → 0 时趋于无穷，并随 ρ 下降（第 2.5 节，数值检验）；
- 对 ρ 二分，求到相邻浮点数，与 `vasicek_fit` 的二分相同；
- 每一步对每一期求一次因子，期数在几十以内时很快。

**报错：**

- 少于两期；
- 某一期不满足 `matrix_factor` 的要求，信息给出该期的位置；
- 各期因子相同，方差为 0，没有 ρ 能使方差为 1；
- ρ 接近 1 时方差仍大于 1。

**Notes 写明：**

- 这是 BSF 的单位方差条件，经 Shao 第 6 页和 Leegstra 第 21 页转述；
- 期数少时估计很不稳定，户数少时偏高；
- 另一条路线：对组合的违约率序列用 `capital.vasicek_fit` 和 `systematic_factor`（Witzany 步骤 1）；
- 方差随 ρ 下降是数值检验的结论。

### 3.4 不做的内容

| 内容 | 理由 |
| --- | --- |
| 每个状态一个相关系数 | Witzany 表 1 显示按等级估计的 ρ 差别很大，但这已不是 BSF 的单参数模型。以后让 `correlation` 也接受每个状态一个值，是兼容的扩充（决定 8） |
| Logistic 等其他分布（Witzany 式 (5)） | 正态是 BSF、CMTD 和 IRB 公式的共同假设；换分布要同时换 ρ 的估计 |
| 因子与宏观变量的回归（Shao 式 (1.7)，Engelmann 式 (6)–(8)） | 属于使用者自己的宏观模型，指南说明 |
| 多因子模型（Shao 第 2 页提到的 Wei 的模型，原文未核对；Leegstra 的主成分做法），以及 Shao 第 3 节直接用宏观变量的模型 | 超出单因子的范围 |
| 平滑长期矩阵（Leegstra 第 20 页） | 平滑方法各异；结构性的 0 在本模型中本来就保持为 0 |
| 按多项分布的极大似然估计因子 | 偏离 BSF 的准则（决定 3） |
| Engelmann 的 TTC 组合 | 一行矩阵运算；指南的“Limits”提醒先不加压力推进一次 |

## 4. 测试与参考

全部放在 `tests/test_lifetime.py`。BSF 原文是否印有可复现的数值无法确认（第 2.0 节），参考值取自 CMTD 和 Witzany。

| 内容 | 对照 |
| --- | --- |
| 阈值（CMTD 表 8.4、8.5） | 把 BB 行放进一个 8 个状态的矩阵，取 ρ = 0.1、因子 0.7，由 `conditional_matrix` 的累计列反推阈值 Φ⁻¹(C)·√(1 − ρ) + √ρ·z，与表 8.5 的 7 个值相差不超过 0.005 |
| Witzany 表 3 | 表 2 各行除以行和，ρ = 0.08，因子 `NormalDist().inv_cdf(0.01)`：72 格与表 3 相差不超过 0.01 个百分点（原型最大 0.009）。第 2.2 节的 6 格超出输入舍入的范围，在测试中列出，并检查其中 5 个违约格与 `conditional_pd` 的结果一致；与 1.5 处理 Pluto–Tasche 表格、1.8 处理 Tasche 表 5 的做法相同 |
| 恒等式 | ρ = 0 时返回 `matrix`，非零格子相对误差 1e-12；对因子取 80 个节点的 Gauss–Hermite 平均，返回 `matrix`，相对误差 1e-12，含 1e-11 的极小格子；各行之和为 1；结构性的 0 和 1 保持不变；违约行为单位行；违约列等于 `conditional_pd`；因子增大时，每行“到 j 或更差”的累计概率都不增加 |
| 蒙特卡罗暴力检验 | 用局部随机数生成器模拟 X = √ρ·z + √(1 − ρ)·Y，按阈值分箱，频率与 `conditional_matrix` 相差在 5 个标准误以内 |
| 标签与顺序 | 结果保留 `matrix` 的索引、列和名称；数组输入得到 `RangeIndex`；违约不在最后时，与把它移到最后、计算、再移回的结果相同 |
| `matrix_factor` 的往返 | 计数取 n_i 乘以 `conditional_matrix(matrix, correlation=ρ, factor=z)` 的行，z 取 −4 到 4，ρ 取 0.01、0.1、0.5：返回 z，绝对误差 1e-9 |
| `matrix_factor` 的暴力检验 | 在随机多项分布的计数上，用 `statistics.NormalDist` 独立实现准则，在 [−8, 8] 上以 0.001 为步长求网格最小点，与结果相差不超过一个步长 |
| `matrix_factor` 与已有函数 | 两状态矩阵等于 `systematic_factor`，相对误差 1e-12；计数乘以 7 不改变结果；整数权重等于重复的行（经 `transition_matrix(weights=)`）；没有户数的行不影响结果 |
| `matrix_correlation` | 结果处各期因子的 `ddof=1` 方差等于 1，误差 1e-9；往返：取一组样本方差恰为 1 的因子 z_t，各期计数取 n_i 乘以 `conditional_matrix(matrix, correlation=ρ, factor=z_t)` 的行，返回 ρ，误差 1e-9；各期计数相同时报错；把各期顺序打乱不改变结果 |
| 报错 | 第 3.1–3.3 节的每一种情形，用 `pytest.raises(..., match=)` |
| 完整流程 | `make_loan_panel` → 按月 `transition_matrix` → `matrix_correlation` → `matrix_factor` → `conditional_matrix` → 乘积 → `convert_pd` |
| SciPy 交叉核对 | 用 `pytest.importorskip("scipy")`：`conditional_matrix` 对照 `scipy.stats.norm` 的写法，相对误差 1e-12；`matrix_factor` 对照 `scipy.optimize.minimize_scalar` 在同一准则上的解，误差 1e-8；`matrix_correlation` 对照 `scipy.optimize.brentq`，误差 1e-8。先放在 `tests/test_lifetime.py`，由集成 PR 移入 `tests/test_reference.py` |

**数据：** CMTD 的 BB 行和表 8.5、Witzany 的表 2 和表 3 写死在测试里，测试本身不依赖 SciPy。

## 5. 指南：`docs/user_guide/lifetime.rst`

**新小节，放在 “Term structures from transition matrices” 末尾：转移矩阵中的经济。**

- 用 `conditional_matrix` 平移指南已有的月度矩阵 `monthly`，打印几个因子下的结果；
- 说明违约列就是 `conditional_pd`，因子为 0 是中位经济，平均才是 `monthly`；
- 用乘积写出多个月的路径，并交给 `convert_pd`。

**新小节：从观察到的矩阵估计因子。**

- **步骤：** 按月用 `transition_matrix(..., states=...)` 得到计数，用 `matrix_correlation` 估计 ρ，用 `matrix_factor` 估计每个月的因子，打印序列和每年的平均（第 2.7 节）。
- **说明三点：**
  - 面板没有经济周期，因子反映的是年轻账簿的账龄变化，所以长期矩阵和各期矩阵应来自可比的组合；
  - 有真实的历史时，再用使用者自己的宏观模型把因子与预测联系起来（Shao 式 (1.7)，Engelmann 式 (6)–(8)）；
  - 另一条路线是对组合违约率用 `vasicek_fit` 和 `systematic_factor`。
- **可选的模拟核对：** 用 `numpy.random.default_rng` 从 `conditional_matrix` 抽出已知因子下的多项分布计数，展示估计能还原这些因子，放在 `.. admonition:: Simulation check` 中。

**改写 “Economic scenarios”：**

- **情景：** 每个情景是一条月度因子路径。保留三个情景和权重；因子 1、0、−1.5 在前 12 个月保持，之后回到 `monthly` 本身（第 2.2 节）。
- **相关系数：** 用上一节估计的 ρ（决定 10）。
- **计算：** 逐月相乘得到每个状态的累计 PD，经 `convert_pd` 得到边际 PD，再算 ECL 和加权 ECL。
- **与现在的区别：** 现在的一节只平移违约概率，升级和降级不变；新的一节平移整个矩阵。
- **输出：** 数值在实现时运行页面重新生成，正文引用的数字随之更新。

**“Limits”：**

- 因子吸收组合结构的变化；
- 因子和 ρ 属于矩阵的期间；
- 所有状态共用一个 ρ，而 Witzany 表 1 显示按等级估计的差别很大；
- 先不加压力推进一次，检查是否出现 Engelmann 所说的虚假周期。

**图：** 不加。如果需要“各月因子”的图，要改 `docs/_plotting.py`，这是共享文件，先问协调会话，或留给集成 PR。

## 6. 文件与流程

| PR | 文件 |
| --- | --- |
| 本稿（第一阶段） | `docs/design/kivi-1.10-matrix-scenarios.md`（新） |
| 实现，审定后开工（第二阶段） | `src/kivi/lifetime.py`、`tests/test_lifetime.py`、`docs/user_guide/lifetime.rst`、`docs/api/lifetime.rst`、`docs/release/unreleased/matrix-scenarios.new.rst`，以及 `kivi.po` |

- **路线图：** `kivi-1.10-roadmap.md` 第 5.2 节由协调会话决定是否改为指向本稿，本会话不改。
- **私有函数：** 沿用 `lifetime.py` 内的 `_transition_probabilities` 和 `_check_transition_rows`；正态函数来自 `_stats`；不导入其他公开模块的私有函数。测试和文档用 `capital.conditional_pd` 和 `systematic_factor` 作对照，实现不依赖它们。
- **集成 PR 需要补的内容**（实现 PR 的描述中会再列一次）：
  - `docs/api/conventions.rst`：因子的符号约定，正值表示经济好于平均，适用于 `capital` 和 `lifetime`；`counts` 是频数权重之和；
  - `docs/api/compatibility.rst`：`conditional_matrix` 的结果保留 `matrix` 的标签；没有新的固定列名；
  - `docs/api/glossary.rst`：转移矩阵的单因子模型、信用周期指数（系统因子）、阈值、中位经济与长期矩阵、最小卡方估计、单位方差条件；
  - 两种语言的 README API 表；
  - SciPy 交叉核对移入 `tests/test_reference.py`；
  - 教程 portfolio 一章是否改用 `conditional_matrix`（路线图第 5.3 节）；
  - CLAUDE.md 中 `lifetime` 的条目：三个函数和本稿。

**流程：**

1. 本稿审定。
2. 实现 PR：从最新的 master 分出，完成后开 ready PR，列出公共 API 和集成 PR 的注意事项。不合并。
3. 集成 PR 和发布 1.10.0 由协调会话按你的指示进行。

## 7. 决定

| # | 问题 | 建议 | 其他选项 |
| --- | --- | --- | --- |
| 1 | 名称 | `conditional_matrix`：参数和公式与 `capital.conditional_pd` 对应，违约列就是它；CERM 第 5.1 节的标题是 “Conditional migration matrices”，Engelmann 称之为给定经济状态的转移矩阵；`matrix_factor`（对应 `systematic_factor`）；`matrix_correlation` | 路线图暂定的 `shift_matrix`；`factor_correlation` |
| 2 | 因子为 0 的含义（确认） | 因子为 0 是中位经济，不是 `matrix`；测试改为“ρ = 0 时返回 `matrix`”和“对因子取平均时返回 `matrix`”；IFRS 9 情景在预测期以后回到 `matrix` | 无：任务说明中“Z = 0 返回原矩阵”的测试在 ρ > 0 时不成立 |
| 3 | 每期因子的估计准则 | BSF 的加权最小二乘：每格的户数除以拟合概率处的二项方差（Shao 式 (1.5)，Leegstra 式 (3.20)） | 按多项分布的极大似然：统计上更规范，但不再是 BSF 的方法，结果在户数少时不同 |
| 4 | ρ 的单位方差口径 | 样本方差 `ddof=1`，均值不强制为 0，与 `vasicek_fit` 相同 | `ddof=0`；或以已知均值 0 计算的二阶矩 Σz²/T = 1 |
| 5 | `matrix_correlation` 的 `counts` | 各期计数矩阵的可迭代对象（列表、元组、`dict.values()`） | 二级索引 (期间, `before`) 的堆叠 DataFrame；三维数组 |
| 6 | 无法估计的情形 | 长期概率为 0 的格子有正计数时报错；没有有限最小点时报错 | 忽略这些格子；返回 ±∞ |
| 7 | `default_state` | 含义与 `transition_pd` 相同，默认最后一个状态；计算阈值时把违约当作最差的状态，其余状态按 `matrix` 的顺序 | 去掉这个参数，总是以最后一个状态为违约 |
| 8 | 每个状态一个 ρ | 不做：BSF 是单参数模型；以后让 `correlation` 也接受每个状态一个值，是兼容的扩充 | 1.10 就接受标量或每个状态一个值（Witzany 的按等级相关系数） |
| 9 | 多年路径 | 指南示范逐期相乘，不改 `transition_pd`，与路线图“逐年相乘”一致 | 让 `transition_pd` 也接受矩阵的序列，每期一个 |
| 10 | 指南 | 按第 5 节：在 `make_loan_panel` 的月度矩阵上估计因子并说明面板没有周期；情景用估计的 ρ 和月度因子路径，预测期以后回到 `monthly`；不加图。另把“给 `make_loan_panel` 加一个系统因子选项，默认输出不变”列为 1.11 的候选，它改 `datasets`，不属于本会话 | 情景保留现在假设的 ρ = 0.1：在月度矩阵上过强，`current` 贷款的 12 个月 PD 从 0.56% 到 19.77% |
