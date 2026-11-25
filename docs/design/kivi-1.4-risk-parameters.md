# kivi 1.4 设计稿：验证报告的检验（含 1.5–1.7 的风险参数规划）

2026-11-23 起草 · 同日按协调会话的分工方案修订：1.4 只做验证检验，与 `docs/design/kivi-1.4-legacy-gaps.md` 并行 · 2026-11-23 审定：第 10 节的 15 项全部采用推荐

## 1. 目标与范围

1.3.0 之后，路线图已经全部完成。本稿把“评分卡之外还适合纳入什么”那次分析里尚未实现的部分，整理成四个版本。它们都沿着巴塞尔风险参数（PD、LGD、EAD）和监管计量往下做，运行时仍只依赖 NumPy 和 pandas。

1.4 同时还有另一份设计稿 `kivi-1.4-legacy-gaps.md`，补 0.2.x 遗留功能的缺口，涉及 `selection.py`、`plotting.py`、`woe.py`。1.1–1.3 每版有三到四个功能 PR，所以本稿在 1.4 只放一个领域，其余顺延。

| 版本 | 主题 | 内容 |
| --- | --- | --- |
| **1.4** | 验证报告的检验 | ECB 验证报告中 kivi 还没有的检验，加 Spiegelhalter 检验和 Brier 分解，按主题放进 `vintage`、`calibration`、`metrics` |
| **1.5** | 监管资本与生存分析 | 新模块 `kivi.capital`：IRB 风险权重函数、Vasicek 单因子模型、低违约组合的 Pluto–Tasche 估计；`kivi.lifetime` 的 Kaplan–Meier 和竞争风险曲线 |
| **1.6** | LGD 和 EAD | 新模块 `kivi.lgd`：清收 LGD、CCF、连续目标的分组表、fractional logit；模拟回收数据 |
| **1.7** | 策略规则与收益 | 新模块 `kivi.rules`：规则命中与规则集瀑布表；`cutoff_table` 的收益列；EMP 指标 |

第 10 节的 15 项已于 2026-11-23 全部采用推荐，其中也包括 1.5–1.7 的事项。1.4 按本稿实现；1.5–1.7 开工前如需调整 API，再提交你审定。

**不做的：**

- 市场风险（VaR、ES 及其回测）：用户群不同，会冲淡 kivi 信用风险的定位。
- 组合信用模型（CreditRisk+、高斯 copula 蒙特卡洛）和经济资本分配：Vasicek 单因子已覆盖主要用途。
- 机器学习模型（GBDT、孤立森林）、图网络反欺诈和 AML：违反“不做模型包装、运行时只依赖 NumPy/pandas”。
- Cox 比例风险模型、多期 Pluto–Tasche、beta 回归、下行 LGD 模型：见各节的“以后再说”。
- 风险定价和 RAROC：口径因机构而异，只在指南里给出由 `expected_loss` 和 `capital_requirement` 组成的计算示例。

## 2. 背景

### 2.1 上次推荐与已实现部分

| 推荐项 | 已在 master 上 | 本稿补上 |
| --- | --- | --- |
| 模型验证 | `roc_auc_interval`、`delong_test`、`roc_table`、`jeffreys_test`、`binomial_test`、`hosmer_lemeshow`、`herfindahl_index` | MWB、迁移矩阵 z 检验、HI/CV 集中度检验、LGD/CCF 的 t 检验与 gAUC、Spiegelhalter 检验、Brier 分解（1.4） |
| 资本计量 | 无（只有 AUC 区间用到的 `_stats.normal_quantile`） | 全部（1.5） |
| 寿命期 PD / IFRS 9 ECL | `kivi.lifetime`：`convert_pd`、`transition_pd`、`expected_credit_loss`、`significant_increase`、`assign_stages`、`stage_table` | 考虑删失的 Kaplan–Meier、提前还款的竞争风险、左截断；多情景前瞻性调整的示例（1.5） |
| LGD / EAD | 无（`expected_credit_loss` 把 LGD 当作输入） | 全部（1.6） |
| 规则分析 | 无（`confusion_table` 只覆盖单个截断点） | 全部（1.7） |
| 收益 | 无 | 收益列、EMP（1.7）；定价只给示例 |

### 2.2 依据的原文

- **ECB, “Instructions for reporting the validation results of internal models – IRB Pillar I models for credit risk”, February 2019。** 1.4 的检验按其中的公式实现，节号见第 3 节。kivi 的 `roc_auc_interval` 已经使用附件 3.1 的 AUC 方差估计（DeLong 的放置值、无偏样本方差），与原文一致。实现前需确认 ECB 是否发布了更新版本。
- **BCBS, Basel Framework CRE31（IRB 风险权重函数）和 CRE32（PD/LGD 下限、有效期限）。** 起草时 BIS 网页上的公式是图片，未能直接转录。第 4.2 节的公式按 2005 年 BCBS《IRB 风险权重函数说明》的通行写法给出，并用 SciPy 复算了公开示例：公司 PD 1%、LGD 45%、M 2.5 年的风险权重为 92.32%。实现 PR 必须逐条对照 CRE31 的公式图片，并在 docstring 的 References 中写明段落号。
- **《商业银行资本管理办法》（国家金融监督管理总局令 2023 年第 4 号，2024 年 1 月 1 日起施行）。** 起草时 gov.cn 的页面无法访问，未能核对内评法附件中的参数（中小企业规模调整的营业收入区间与单位、PD 和 LGD 下限、是否有 1.06 缩放因子）。因此 API 把这些参数都做成关键字，实现 PR 对照附件原文，在指南中列出中国的取值。

### 2.3 设计约束（来自 CLAUDE.md 和 1.4 的分工方案）

- 公共模块不能导入另一个公共模块的私有函数，多个模块共用的数值函数放在 `_stats`。
- `tests/test_validation.py` 已被输入校验的测试占用，所以新模块不能叫 `kivi.validation`。
- 两个进行中的 PR 不能改同一个文件。1.4 中 `selection.py`、`plotting.py`、`woe.py` 及其测试和指南归 legacy-gaps 的功能 PR 所有，本稿的 PR 不碰。
- 功能 PR 和设计 PR 都不改共享入口：README（两种语言）、`docs/index.rst`、`user_guide/index.rst` 和 `api.rst` 的顺序、`compatibility.rst`、`glossary.rst`、`conventions.rst`、`migration.rst`、CLAUDE.md，以及 `tests/test_validation.py` 的登记表。各 PR 在描述中列出需要补的内容，由协调会话在集成 PR 中统一补上，合并和发布也由协调会话按你的指示进行。`kivi.po` 是唯一允许每个 PR 都改的共享文件。
- `kivi.plotting` 的新图要等 legacy-gaps 的图表 PR 合并之后再做。

## 3. 1.4：验证报告的检验

### 3.1 放在哪里

按 kivi 一贯的做法，按主题放进现有模块，而不是按监管机构建模块：

- 迁移类放在 `transition_matrix` 所在的 `kivi.vintage`；
- 集中度、回测和校准放在 `herfindahl_index` 和 `jeffreys_test` 所在的 `kivi.calibration`；
- gAUC 放在 `roc_auc_interval` 所在的 `kivi.metrics`。

新增的用户指南 `docs/user_guide/validation.rst` 按 ECB 报告的顺序，把每个工具对应到 kivi 函数，已有的 `jeffreys_test`、`roc_auc_interval` 和 `herfindahl_index` 也列进去。

| ECB 节号 | 工具 | kivi |
| --- | --- | --- |
| 2.5.3.1 | PD 回测：Jeffreys 检验 | 已有 `jeffreys_test` |
| 2.5.4.1、附件 3.1 | 当前 AUC 对初始 AUC | 已有 `roc_auc_interval` 给出估计和标准误；S = (AUC_init − AUC_curr)/s，p = 1 − Φ(S)，指南给出两行示例 |
| 2.5.5.1 | 客户迁移：上、下 MWB | 新增 `migration_bandwidth` |
| 2.5.5.2 | 迁移矩阵稳定性：z 检验 | 新增 `migration_test` |
| 2.5.5.3 | 等级集中度：HI 与 CV 检验 | 新增 `concentration_test` |
| 2.6.2.1、2.7.2.1、2.8.2.1、2.9.3.1 | LGD、ELBE、LGD in-default、CCF 的 t 检验 | 新增 `backtest_table` |
| 2.6.3.1、2.9.4.1、附件 3.2 | LGD/CCF 的 gAUC（Somers' D） | 新增 `generalized_auc` |
| — | Spiegelhalter 检验 | 新增 `spiegelhalter_test` |
| — | Brier 分解 | 新增 `brier_decomposition` |

### 3.2 API

```python
# kivi.vintage
migration_bandwidth(before, after, *, grades, weights=None)
    # -> Series，索引 statistic：upper、lower
migration_test(before, after, *, grades, weights=None)
    # -> DataFrame，二级索引 (before, after)，每个非对角格一行：total、share、statistic、p_value

# kivi.calibration
concentration_test(counts, *, initial_counts=None, initial_cv=None)
    # -> Series，索引 statistic：hi、cv、initial_hi、initial_cv、statistic、p_value
backtest_table(groups, realized, estimated, *, weights=None)
    # -> DataFrame，索引 bin：total、estimated、realized、difference、variance、statistic、p_value
spiegelhalter_test(probability, target, *, weights=None)
    # -> (statistic, p_value)，与 hosmer_lemeshow 一样返回元组
brier_decomposition(groups, target, probability, *, weights=None)
    # -> Series，索引 statistic：brier、reliability、resolution、uncertainty、
    #    within_variance、within_covariance

# kivi.metrics
generalized_auc(estimated, realized, *, level=0.95, weights=None)
    # -> DataFrame，行 gauc、somers_d；列 estimate、se、lower、upper（同 roc_auc_interval）
```

### 3.3 公式与语义

**迁移（ECB 2.5.5.1–2.5.5.2）。** `grades` 是从好到差的 K 个评级。N_i 是期初在等级 i 的客户数，其中包括期末违约、退出或换模型的客户；N_ij 是期末在等级 j 的客户数；p_ij = N_ij / N_i。`after` 中不属于 `grades` 的值（如 `"default"`、`"closed"`）只计入 N_i。`before` 必须属于 `grades`，缺失值报错，需要先把退出的客户标为某个标签。

```
upper MWB = Σ_{i=1}^{K−1} Σ_{j>i} |i − j|·N_i·p_ij / M_u，  M_u = Σ_{i=1}^{K−1} max(|i − K|, |i − 1|)·N_i·Σ_{j>i} p_ij
lower MWB = Σ_{i=2}^{K}   Σ_{j<i} |i − j|·N_i·p_ij / M_l，  M_l = Σ_{i=2}^{K}   max(|i − K|, |i − 1|)·N_i·Σ_{j<i} p_ij

z_ij = (p_{i,j+1} − p_ij) / √([p_ij(1 − p_ij) + p_{i,j+1}(1 − p_{i,j+1}) + 2·p_ij·p_{i,j+1}] / N_i)    j < i
z_ij = (p_{i,j−1} − p_ij) / √([p_ij(1 − p_ij) + p_{i,j−1}(1 − p_{i,j−1}) + 2·p_ij·p_{i,j−1}] / N_i)    j > i
p 值 = Φ(z_ij)；分母为 0 或 N_i = 0 时 statistic 和 p_value 为 NaN（ECB 报告为空）
```

**集中度（ECB 2.5.5.3）。** R_i 是 K 个等级的份额（按客户数或敞口），空等级也算在 K 里。

```
CV = √(K·Σ(R_i − 1/K)²)，  HI = 1 + ln((CV² + 1)/K) / ln K = 1 + ln(HHI) / ln K
p 值 = 1 − Φ(√(K − 1)·(CV − CV_init) / √(CV²·(0.5 + CV²)))
```

`HHI` 是已有 `herfindahl_index` 的结果，HI 是它的对数归一化（均匀时为 0，全部集中在一个等级时为 1），指南说明两者的换算。`initial_counts` 和 `initial_cv` 恰好给一个；只算敞口加权的 HI 时可以都不给，此时 statistic 和 p_value 为 NaN。

**回测 t 检验（ECB 2.6.2.1 等）。** 每组 d_i = realized_i − estimated_i：

```
T = √N·mean(d) / s，  s² = Σ(d_i − mean(d))² / (N − 1)，  p 值 = 1 − t_{N−1}(T)
```

这是单侧检验，原假设为估计值不低于真实值，p 值小说明低估。`realized` 和 `estimated` 是连续值，不受二分类 `target` 的约定限制。组的标签照原样计数，与 `calibration_table` 相同；整体组合层面的检验传一个常数组即可。N = 1 时方差无定义，结果为 NaN。指南给出 ECB 的 12 个 LGD 分段：

```python
pd.cut(estimated, [0, .05, .1, .2, .3, .4, .5, .6, .7, .8, .9, 1, np.inf], right=False)
```

**gAUC（ECB 附件 3.2）。** `estimated` 和 `realized` 是已经离散化的有序分段（有序分类按类别顺序，数值按大小），行是估计分段，列是实际分段，列联表为 a_ij。

```
A_ij = Σ_{k<i,l<j} a_kl + Σ_{k>i,l>j} a_kl，  D_ij = Σ_{k>i,l<j} a_kl + Σ_{k<i,l>j} a_kl
P = Σ a_ij·A_ij，  Q = Σ a_ij·D_ij，  w_r = F² − Σ r_i²
D(C|R) = (P − Q) / w_r，  gAUC = (D(C|R) + 1) / 2
s(gAUC) = √(Σ a_ij·(w_r·d_ij − (P − Q)·(F − r_i))²) / w_r²，  d_ij = A_ij − D_ij
```

原文的 s 就是 gAUC 的标准误。Brown 和 Benedetti（1977）的 Somers' D 渐近标准误比它多一个因子 2，而 gAUC = (D + 1)/2，所以两者恰好相差一半。起草时做了模拟核对：每次 400 个样本、4×3 分段、重复 4000 次，gAUC 的实际标准差为 0.0175，原文 s 的均值为 0.0177。表中 `somers_d` 一行的标准误取 2s。D(C|R) 等于 SciPy 的 `somersd(estimated, realized).statistic`。当前对初始的检验与 AUC 相同，写在指南里。

**Spiegelhalter（1986）。** 双侧检验，加权时各项乘以权重：

```
z = Σ(y_i − p_i)(1 − 2p_i) / √(Σ(1 − 2p_i)²·p_i(1 − p_i))
```

**Brier 分解（Murphy 1973；Stephenson 等 2008）。** 按 `groups` 分组，f̄_k 和 ō_k 是组内的平均预测和事件率，下式恒等成立。组内预测相同（如等级 PD）时，后两项为 0，即 Murphy 的三项分解。

```
brier = reliability − resolution + uncertainty + within_variance − within_covariance
reliability = Σ n_k(f̄_k − ō_k)² / N，     resolution = Σ n_k(ō_k − ō)² / N，  uncertainty = ō(1 − ō)
within_variance = Σ_k Σ_{i∈k} (f_i − f̄_k)² / N，  within_covariance = 2·Σ_k Σ_{i∈k} (f_i − f̄_k)(o_i − ō_k) / N
```

### 3.4 `_stats` 的新增函数

1.4 的 PR 拥有 `_stats.py`，只加本版需要的：

- `normal_cdf`，支持数组（z 检验和 gAUC 区间用）；
- `student_t_sf`，由已有的 `betainc` 计算（t 检验用）。

现有的 `normal_quantile` 不变。1.5 的资本计量再加数组版分位函数、`beta_quantile` 和 `bivariate_normal_cdf`。

### 3.5 图与以后再说

- **图：** 1.4 不新增 `kivi.plotting` 函数。迁移矩阵热力图如有需要，等 legacy-gaps 的图表 PR 合并之后再做。指南中的示意图按惯例由功能 PR 在描述中写明，在集成 PR 中加入 `docs/_plotting.py`。
- **以后再说：** 多期正态检验（Blochwitz 等 2004）、带相关的二项检验、红绿灯法。它们都能由现有函数加 1.5 的 `conditional_pd` 组合出来，等有实际需求再加。

## 4. 1.5 A：`kivi.capital`

### 4.1 API

```python
asset_correlation(probability, *, asset_class, turnover=None,
                  turnover_range=(5.0, 50.0), financial=False)
    # -> 资产相关系数 R，形状同输入，Series 名为 "correlation"

capital_requirement(probability, lgd, *, asset_class, maturity=None, turnover=None,
                    turnover_range=(5.0, 50.0), financial=False, correlation=None,
                    level=0.999, scaling=1.0)
    # -> 资本要求 K（占 EAD 的比例），Series 名为 "capital_requirement"
    #    风险加权资产 RWA = 12.5 * K * EAD，由调用者相乘

conditional_pd(probability, *, correlation, factor=None, level=None)
    # -> 给定系统因子时的条件 PD；factor 与 level 二选一

systematic_factor(default_rate, probability, *, correlation)
    # -> 由观察到的违约率反推系统因子 z（conditional_pd 的逆运算）

vasicek_fit(default_rate)
    # -> (probability, correlation)：由一段违约率时间序列估计长期 PD 和相关系数

most_prudent_pd(event, total, *, level=0.95, correlation=0.0)
    # -> 每个等级的 PD 上界（Pluto–Tasche），等级从风险最低排到最高
```

`asset_class` 取 `"corporate"`（公司、主权、银行）、`"mortgage"`（个人住房抵押）、`"revolving"`（合格循环零售）或 `"other_retail"`（其他零售）。

### 4.2 公式

记 N 为标准正态分布函数，G 为其分位函数。

```
公司：  w = (1 − exp(−50·PD)) / (1 − exp(−50))
        R = 0.12·w + 0.24·(1 − w)
        中小企业：R −= 0.04·(1 − (S − lo) / (hi − lo))，S 先截到 [lo, hi]；S ≥ hi 不调整
        大型或无监管金融机构（financial=True）：R ×= 1.25
        b = (0.11852 − 0.05478·ln PD)²
        K = [LGD·N((G(PD) + √R·G(level)) / √(1 − R)) − PD·LGD] · (1 + (M − 2.5)·b) / (1 − 1.5·b)
零售：  个人住房抵押 R = 0.15；合格循环零售 R = 0.04
        其他零售：w = (1 − exp(−35·PD)) / (1 − exp(−35))，R = 0.03·w + 0.16·(1 − w)
        K = LGD·N((G(PD) + √R·G(level)) / √(1 − R)) − PD·LGD    （无期限调整）
最后：  K ×= scaling
```

**参数与边界：**

- `turnover_range=(lo, hi)` 默认为巴塞尔的 (5, 50)（百万欧元）。中国办法的区间和单位由实现 PR 核对后写入指南。不是中小企业的借款人传 `turnover=np.inf`，缺失值报错，不把“未知”当作“不调整”。只有公司类接受 `turnover` 和 `financial`。
- `maturity`：公司类默认 2.5 年，必须为正；不自动截到 [1, 5]，因为短期敞口有例外，期限下限由调用者按所在辖区规则处理。零售类传入 `maturity` 报错。
- `PD` 必须严格在 0 和 1 之间。下限不自动施加，由调用者先 `np.maximum(probability, floor)`，指南列出巴塞尔的取值（CRE32.4：除主权外不低于 0.05%）。违约敞口（PD = 1）报错，提示按 CRE31.3 计算 `max(0, LGD − ELBE)`。
- `correlation` 给定时替代公式中的 R（用于经济资本或其他口径），`asset_class` 仍决定是否做期限调整。`level` 是单因子模型的置信水平，默认 0.999；经济资本可用 0.9997 等。
- `scaling` 默认 1.0。巴塞尔 II 曾对信用风险加权资产乘以 1.06，现行框架和中国办法是否保留由实现 PR 核对后写入指南。

### 4.3 Vasicek 单因子模型

```
conditional_pd：   PD(z) = N((G(PD) − √ρ·z) / √(1 − ρ))
                   level=q 等价于 z = −G(q)，与 capital_requirement 中的项一致
systematic_factor：z_t = (G(PD) − √(1 − ρ)·G(DR_t)) / √ρ
```

- **符号：** `factor` 为正表示经济好于平均，条件 PD 低于长期 PD。这是 Merton 模型中资产收益的方向。
- **`vasicek_fit`** 用违约率本身的矩估计：PD 取样本均值 m，ρ 解方程 Φ₂(G(m), G(m); ρ) − m² = 样本方差（无偏，ddof=1）。左边随 ρ 单调，用二分法求解。它允许某些期的违约率为 0，而对违约率做 probit 变换的方法在违约率为 0 时无法计算。样本方差为 0 时 ρ = 0；方差不小于 m(1 − m) 时报错。
- **`systematic_factor`** 遇到 0 或 1 的违约率报错，因为 G(0) 无界。
- **IFRS 9 前瞻性调整：** 在 lifetime 指南中给出示例（见第 5 节）：
  1. 对每个情景的因子路径 z_t，用 `conditional_pd` 把 TTC 的条件 PD 曲线转成 PIT 曲线；
  2. 经 `convert_pd` 转为边际 PD 后传给 `expected_credit_loss`；
  3. 按情景权重加总。

### 4.4 Pluto–Tasche 最审慎估计

等级 k 的上界用该等级及所有更差等级的合计：n_k = Σ_{i≥k} N_i，d_k = Σ_{i≥k} D_i。

- **独立（`correlation=0`）：** p_k 满足 P(Bin(n_k, p_k) ≤ d_k) = 1 − level，即 Beta(d_k + 1, n_k − d_k) 分布的 `level` 分位数。d_k = n_k 时为 1。零违约时为 1 − (1 − level)^{1/n_k}。
- **相关（`correlation=ρ`）：** 违约在给定系统因子时独立，P(D ≤ d) = ∫ P(Bin(n, PD(z)) ≤ d) φ(z) dz。积分用 Gauss–Hermite 求积（NumPy 的 `hermegauss`），对 p 二分求解。
- **以后再说：** 多期并带系统因子自相关的版本需要蒙特卡洛。
- **参考值：** 原论文的零违约算例，N = (100, 400, 300)、level 0.9，给出 0.29%、0.33%、0.76%（arXiv:cond-mat/0411699v2 表 1–3；1 − 0.1^(1/300) = 0.7646%，起草时误记为 0.77%）。

### 4.5 `_stats` 的新增函数

1.5 的资本会话先单独提交一个只改 `_stats.py` 的小 PR：

- 数组版 `normal_quantile`（标量调用照旧）；
- `beta_quantile`：对 `betainc` 二分或 Newton；
- `bivariate_normal_cdf`：Plackett 恒等式加 Gauss–Legendre 求积。

## 5. 1.5 B：生存分析（`kivi.lifetime`）

### 5.1 API

```python
survival_table(duration, event, *, competing=None, entry=None, weights=None)
    # -> DataFrame，索引 period（1 到最大 duration，每期一行）：
    #    at_risk、event、[competing]、censored、hazard、survival、cumulative_pd
```

- `duration`：退出或最后观察到的期数，为正整数，例如月账龄。
- `event`：在该期违约为 1，否则为 0。
- `competing`：在该期因其他原因退出为 1，例如提前结清；与 `event` 同时为 1 报错。
- `entry`：开始观察前已经过的期数，默认 0。观察窗口晚于放款时（如数据从 2024 年 1 月开始），用它做左截断，账户从 entry + 1 期才计入风险集。

### 5.2 公式

同一期内的删失排在违约之后，即删失的账户在该期仍在风险集中。

```
n_t = Σ w·1[entry < t ≤ duration]；d_t、c_t 为该期的违约和竞争退出
hazard_t = d_t / n_t
survival(t) = Π_{s≤t} (1 − (d_s + c_s) / n_s)                      所有原因
不给 competing：cumulative_pd(t) = 1 − Π_{s≤t} (1 − d_s / n_s)      Kaplan–Meier
给 competing：  cumulative_pd(t) = Σ_{s≤t} survival(s − 1)·d_s / n_s   Aalen–Johansen 累计发生率
```

`cumulative_pd` 可以直接传给 `convert_pd(kind="cumulative", to="marginal")`。存在提前还款时，把还款当作删失的 1 − KM 会高估寿命期 PD，所以指南推荐用累计发生率。风险集为 0 之后的期为 NaN。

**不提供置信区间。** Greenwood 方差只适用于 KM，Aalen–Johansen 的方差更复杂，等有需求再加。

### 5.3 数据与文档

- **`make_loan_panel` 新增可选的 `prepayment=0.0`：** 当前贷款每月提前结清的概率。默认值的输出与之前完全相同。结清当月余额为 0、此后没有记录，因此“余额为 0 且账龄小于期限”就是提前结清，不增加列，符合 `compatibility.rst` 中数据集列和 dtype 不变的承诺。
- **指南：** 给出从逐月面板求 `duration`、`event`、`competing` 的示例，并与完整观察的批次的 `vintage_table` 对照；离散时间风险模型用“人-期”展开加 `logistic_regression`，只写示例，不新增函数。
- **多情景前瞻性调整：** 需要资本会话的 `conditional_pd`。若资本 PR 先合并，由生存分析会话写进 lifetime 指南，否则放进 1.5 的集成 PR。

## 6. 1.6：`kivi.lgd`

```python
workout_lgd(account, default_date, ead, flow_account, flow_date, flow_amount, *, rate)
    # -> 每个账户的清收 LGD，Series 名为 "lgd"
conversion_factor(ead, drawn, limit)
    # -> CCF = (EAD − drawn) / (limit − drawn)，limit == drawn 时为 NaN；Series 名为 "ccf"
lgd_table(groups, lgd, *, weights=None)
    # -> DataFrame，索引 bin：total、total_share、mean、std、min、max、zero_share、full_share
fractional_regression(X, response, *, weights=None)
    # -> 与 logistic_regression 相同的系数表，标准误为稳健（HC0）标准误
```

- **`workout_lgd`：** 参数顺序与 `delinquency_panel` 相同，先是账户，后是现金流。

  ```
  LGD_i = 1 − Σ_j flow_j / (1 + rate)^{(t_j − t_D)/365} / EAD_i
  ```

  - 回收为正，费用为负；违约日之前的现金流报错，没有现金流的账户 LGD 为 1。
  - 结果不截断到 [0, 1]：回收超过 EAD 时为负，费用较高时可超过 1，是否截断由调用者按政策决定。
  - 未结案的回收不做外推，回测只用已结案的账户（ECB 2.6.1）。
- **`lgd_table`：** 连续目标版的 `woe_table`，用于筛选 LGD 的驱动因素。`zero_share` 和 `full_share` 是 LGD ≤ 0 和 ≥ 1 的份额，用来看双峰分布。分组仍用 `bin_edges`；按方差下降的有监督分箱会改动 `binning.py`，届时再定。
- **`fractional_regression`：** Papke 和 Wooldridge（1996）的拟似然 logit，用已有的 `_logit` 迭代加权最小二乘拟合 [0, 1] 内的连续响应。分离检测要改写，因为连续响应没有“类”。标准误用夹心估计，参考 statsmodels 的 `GLM(Binomial).fit(cov_type="HC0")`。
- **`kivi.datasets.make_recoveries`：** 模拟违约账户的 EAD、押品、分段和清收现金流，LGD 呈双峰分布，供指南和测试使用。
- **验证：** 1.4 的 `backtest_table` 和 `generalized_auc` 直接用于 LGD 和 CCF。

## 7. 1.7：策略规则与收益

```python
# kivi.rules（新模块）
rule_table(rules, target, *, weights=None)
    # -> DataFrame，索引 rule：hits、hit_rate、event、event_rate、lift、recall、unique_hits、unique_event
rule_set_table(rules, target, *, weights=None)
    # -> DataFrame，索引 rule，按列的顺序依次执行：new_hits、new_event、declined、recall、remaining_event_rate

# kivi.strategy
cutoff_table(score, target, *, cutoffs=10, loss=None, revenue=None, weights=None)
    # revenue 给定时增加 approved_revenue；与 loss 都给时增加 profit = approved_revenue − approved_loss

# kivi.metrics
expected_maximum_profit(score, target, *, p0=0.55, p1=0.1, roi=0.2644, weights=None)
    # -> Series，索引 statistic：emp、rejected_share
```

- **`rules`：** 每列一条规则，值为布尔，命中为 True。`lift` = 命中样本的事件率 / 总体事件率；`unique_hits` 是只被这一条规则命中的样本数。`rule_set_table` 的每一行只计入前面的规则没有命中的样本。规则的重叠矩阵用 `rules.T @ rules` 即可，指南给出示例。
- **目标：** 必须是 0/1，与其他函数相同。拒绝样本可以先做拒绝推断，再通过 `weights` 传入。
- **EMP：** 按 Verbraken 等（2014）的信用评分版本。
  - 对拒绝坏客户的收益 λ（即 LGD）取分布：在 0 处有点质量 p0，在 1 处有点质量 p1，其余在 (0, 1) 上均匀；误拒好客户的损失为 ROI。
  - 在 ROC 凸包上对 λ 积分，得到期望最大利润及对应的拒绝比例。
  - 默认值取自原论文。参考值用 R 的 `EMP` 包或 Python 的 `empulse` 在临时环境中算出后写入测试，不增加依赖。
- **以后再说：** 用多变量决策树挖掘规则。需要一棵新的多变量树，规模较大。

## 8. 共同约定

- **参数顺序：** 沿用 `conventions.rst`。新名字有：`realized`、`estimated`（回测和 gAUC）、`grades`（迁移）、`asset_class`、`duration`、`event`、`competing`、`entry`、`response`、`rules`。名字和取值都要在集成 PR 里写进 glossary 和 conventions。
- **结果形状：**
  - 一值对一值的结果形状同输入，用 `like_input` 加 overload 声明。
  - 表格用 DataFrame，命名统计量用以 `statistic` 为索引的 Series 或 DataFrame。
  - 供另一个函数使用的参数用元组，如 `vasicek_fit` 返回的两个数直接传给 `conditional_pd`。
- **缺失值：**
  - 逐值换算的函数对缺失的 PD、违约率或金额返回 NaN：`asset_correlation`、`capital_requirement`、`conditional_pd`、`systematic_factor`、`conversion_factor`。决定公式分支的参数例外，如缺失的 `turnover` 报错（第 4.2 节）。
  - 计数、检验和拟合的函数遇到缺失值报错：其余全部。
  - 这与现有规则一致。
- **样本权重：** 计数类函数都接受 `weights`，作为频数权重。`vasicek_fit` 和 `most_prudent_pd` 的输入本身就是计数或比率，不接受权重。
- **布尔值：** `probability`、`target` 中的布尔值按 0/1 计；`lgd`、`ead`、`turnover`、`maturity`、金额和 `realized`/`estimated` 拒绝布尔值。

## 9. 验证、测试与分工

### 9.1 参考实现

| 内容 | 对照 |
| --- | --- |
| MWB、迁移 z 检验、集中度、t 检验 | 小矩阵手算；t 检验对照 `scipy.stats.ttest_1samp(alternative="greater")` |
| `generalized_auc` | Somers' D 对照 `scipy.stats.somersd(estimated, realized)`；标准误手算，并用模拟核对 |
| `spiegelhalter_test`、`brier_decomposition` | 手算；分解恒等式；组内预测相同时等于 Murphy 分解 |
| `_stats` 新函数 | SciPy 的 `norm`、`t`，以及 1.5 的 `beta.ppf` 和 `multivariate_normal.cdf` |
| `capital_requirement`、`asset_correlation` | 对 PD、LGD、M、S、各资产类别的网格，与用 `scipy.stats.norm` 独立写出的公式比对到 1e-12；固定值 92.32% |
| `conditional_pd`、`systematic_factor` | 互为逆运算；`level` 与 `factor` 的换算 |
| `vasicek_fit` | 用 `conditional_pd` 模拟的长序列恢复参数；方程两边与 SciPy 比对 |
| `most_prudent_pd` | 零违约的闭式解和原论文算例；相关情形用 `scipy.integrate.quad` |
| `survival_table` | statsmodels 的 `SurvfuncRight`（支持 `entry` 和 `freq_weights`）和 `CumIncidenceRight`（不支持 `entry`，左截断下的累计发生率手算）；完整批次与 `vintage_table` 一致 |
| `fractional_regression` | statsmodels `GLM(Binomial)`，`cov_type="HC0"` |
| EMP | R `EMP` 或 `empulse` 的固定值 |

SciPy 和 statsmodels 已在 `test` extra 里，不增加依赖。依照“一个文件只属于一个进行中的 PR”，功能 PR 把交叉验证先写在自己的测试文件里，并用 `pytest.importorskip`；集成 PR 再移入 `tests/test_reference.py`，并补齐 `test_validation.py` 和 `test_typing.py` 的登记。

### 9.2 1.4 的会话与文件

一个会话 `validation-tests`，可以分成一到两个 PR，由该会话视工作量决定。

| 拥有的文件 | 说明 |
| --- | --- |
| `src/kivi/_stats.py` | `normal_cdf`、`student_t_sf` |
| `src/kivi/calibration.py`、`tests/test_calibration.py` | `concentration_test`、`backtest_table`、`spiegelhalter_test`、`brier_decomposition` |
| `src/kivi/metrics.py`、`tests/test_metrics.py` | `generalized_auc` |
| `src/kivi/vintage.py`、`tests/test_vintage.py` | `migration_bandwidth`、`migration_test` |
| `docs/user_guide/validation.rst`（新）、`calibration.rst`、`metrics.rst`、`vintage.rst` | 新指南和三个现有指南的相关小节 |
| `docs/api/calibration.rst`、`metrics.rst`、`vintage.rst` | 列出新函数 |
| `docs/release/unreleased/validation-tests.new.rst` | 发布说明片段 |

这些文件与 legacy-gaps 的功能 PR（`selection.py`、`plotting.py`、`woe.py` 及其测试和指南）互不重叠。PR 描述中列出公共 API，以及集成 PR 要补的登记、术语、兼容性条目和示意图。

### 9.3 以后版本的文件（供排期参考）

| 版本 | 会话 | 文件 |
| --- | --- | --- |
| 1.5 | `capital` | `_stats.py`（先单独一个小 PR）、`capital.py`（新）、`tests/test_capital.py`、`capital.rst` |
| 1.5 | `survival` | `lifetime.py`、`datasets/_loan_panel.py`、`tests/test_lifetime.py`、`tests/test_datasets.py`、`lifetime.rst` 和 `datasets.rst` 的相关部分 |
| 1.6 | `lgd` | `lgd.py`（新）、`datasets/_recoveries.py`（新），可能需要改 `_logit.py` 的分离检测 |
| 1.7 | `rules`、`profit` | `rules.py`（新）；`strategy.py`、`metrics.py` |

### 9.4 流程

1. 本稿审定：2026-11-23 已确认，第 10 节的 15 项全部采用推荐。
2. 两份 1.4 设计稿都审定后，`validation-tests` 会话开工，与 legacy-gaps 的功能 PR 并行。
3. 评审后，由协调会话按你的指示合并，并写 1.4 的集成 PR。
4. 发布 1.4.0，需要你的确认。
5. 1.5–1.7 开工前再审定各自的 API，按同样的流程进行。

## 10. 决定

2026-11-23，以下 15 项全部采用推荐方案。第 1–5 项用于 1.4，其余用于 1.5–1.7。

| # | 问题 | 决定 | 未采用的选项 |
| --- | --- | --- | --- |
| 1 | 版本拆分 | 1.4 验证检验；1.5 资本与生存分析；1.6 LGD/EAD；1.7 规则与收益 | 1.4 同时做资本、验证、生存（三个会话，与 legacy-gaps 合计六七个 PR）；全部放进 1.4 |
| 2 | 验证检验放在哪里 | 按主题放进 `vintage`、`calibration`、`metrics`，另加 validation 指南 | 新模块 `kivi.backtesting` |
| 3 | “当前 AUC 对初始 AUC”检验 | 指南示例，用 `roc_auc_interval` 的标准误 | 单独的函数 |
| 4 | 迁移函数的输入 | 原始的 `before`、`after` 加 `grades`，与 `transition_matrix` 一致，退出和违约计入 N_i | 接受 `transition_matrix` 的计数表 |
| 5 | Brier 分解的形式 | 五项恒等分解（含组内两项），组内预测相同时退化为 Murphy 三项 | 只做 Murphy 三项 |
| 6 | 资本模块名 | `kivi.capital` | `kivi.irb` |
| 7 | 辖区参数（PD/LGD 下限、SME 区间、1.06） | 不内置辖区预设，做成关键字（默认巴塞尔），指南列出巴塞尔和中国的取值 | `jurisdiction="basel"/"china"` 预设 |
| 8 | 系统因子的符号 | `factor` 为正表示好于平均 | 为正表示压力 |
| 9 | `vasicek_fit` 的方法 | 违约率的矩估计（允许 0 违约率） | probit 变换后的矩估计；极大似然 |
| 10 | `survival_table` 的范围 | 含竞争风险（Aalen–Johansen）和左截断 `entry` | 只做 Kaplan–Meier |
| 11 | 提前还款数据 | `make_loan_panel(prepayment=0.0)`，默认输出不变 | 指南用手工数据 |
| 12 | 清收 LGD 是否截断 | 不截断，负值和大于 1 的值原样返回，日计数 ACT/365 | 截断到 [0, 1] |
| 13 | `fractional_regression` 的标准误 | 稳健（HC0）标准误 | 模型标准误 |
| 14 | 规则分析的位置 | 新模块 `kivi.rules` | 并入 `kivi.strategy` |
| 15 | 定价与 RAROC | 只给指南示例 | 提供函数 |
