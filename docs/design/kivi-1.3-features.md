# kivi 1.3 设计稿：批量时间窗特征（`kivi.features`）

2026-11-22 起草 · 2026-11-23 审定：第 8 节的 9 项全部采用推荐 · 2026-11-23 按 PR #72 的评审决定修订列名大小写、窗口对、`days_since_last` 和 `FeatureSpec` 的 JSON 结构

## 目标与范围

1.3 新增模块 `kivi.features`。它对交易、还款、查询等事件数据，按主体和观察日计算“近 N 个月”的聚合特征。计算只用观察日之前的数据，特征定义可以存成 JSON，保证开发和生产算出同样的数。运行时仍只依赖 NumPy 和 pandas。

**要解决的问题。** 评分卡的行为类变量大多来自事件流水，比如近 3 个月交易笔数、近 12 个月最大逾期天数、距上次逾期的月数。手写的 `groupby` 代码容易用到观察日之后的数据（信息穿越）。开发脚本和生产代码又往往各写一套，结果对不上。

**1.3 要做的：**

- **时间窗聚合：** 每个（主体，观察日）一行，对每个窗口、每个数值列计算计数、求和、均值、最值等。
- **衍生特征：** 窗口之间的比率，以及近期相对前一期的变化（环比、同比）。
- **还款行为特征：** 从逐月状态面板（`delinquency_panel` 或 `make_loan_panel` 的输出）计算最长连续逾期月数、距上次逾期的月数等。
- **`FeatureSpec`：** 仿照 `BinningSpec`，是一个不可变、可存为 JSON 的特征定义，开发和生产共用。

**不做的：**

- Spark 和 SQL 版本：你之前已决定不做。
- 自动穷举生成特征，如 featuretools 的深度特征合成。
- 在线特征存储。
- 群体内百分位排名，即 0.2.x 的行业排名，可以放到以后的版本。

特征筛选沿用已有的 `profile_table`、`correlation_filter` 和 `stepwise`。

## 背景

0.2.x 有三套特征衍生代码，都绑定 Spark 和 Hive，没有统一的防穿越规则。它们的做法和 1.3 的对应如下。

| 0.2.x 代码 | 做了什么 | 1.3 对应 |
| --- | --- | --- |
| `FeatureTools/FeatureTools.py` | 按 uuid 的逐月还款记录，在近 N 个月的窗口内算聚合、连续还款月数、未还月数；比率、中位数交叉、缺失交叉特征 | 时间窗聚合、还款行为特征、窗口比率；交叉特征不做 |
| `FeatureTools/FeatureToolsV2.py` | 对一个快照日取 3/6/9/12 个月窗口，算 cnt、sum、avg、去重计数、最值、偏度等，转成长表写入 Hive；对一段日期回溯跑批；行业百分位排名 | 时间窗聚合；观察表里放多个日期即是回溯；百分位排名不在 1.3 |
| `index/collect.py` | 用 `IndexField` 配置库表、列、算子、回溯月数和主体层级，算基础指标、环比、同比，写长表 | `FeatureSpec` 承担“配置”的角色；环比、同比作为衍生特征 |

kivi 现有可以衔接的功能：

- `delinquency_panel` 和 `make_loan_panel` 产出逐月状态面板，是还款行为特征的输入。
- `make_applications` 给出带申请日期的申请样本，可以作为观察表。
- `profile_table`（IV、KS、卡方检验）、`correlation_filter` 和 `stepwise` 用于筛选衍生出来的大量特征。
- `FeatureSpec` 照 `BinningSpec` 的做法实现：不可变、带版本号的 JSON。

## 数据模型与防穿越规则

一个特征值只由观察日之前、窗口之内的事件决定。默认窗口是左闭右开的 [t − k, t)，观察日 t 当时及以后的事件一律不用。

```
                       窗口 6M_3M（前一期）        窗口 3M（近期）
                    |<----------------------->|<----------------------->|
时间 ──●────────────[──────●─────────●─────────[────●───────●────────────)──●───●──>
       ^            ^                         ^                         ^
       12 个月前的事件  t − 6 个月               t − 3 个月                  t：观察日
       （3M、6M_3M 窗口都不用；                                         t 当时及以后的事件不用
        12M 窗口会用）
```

**两张输入表：**

- **事件表 `events`：** 每行一个事件，含主体列（如 `customer_id`）、时间列和若干数值列（如 `amount`）。不要求排序，同一时刻可以有多行。
- **观察表 `observations`：** 每行一个（主体，观察日），如申请样本的客户和申请日。同一主体可以有多个观察日，即回溯跑批。输出与观察表同索引、同行序，可以直接拼回建模样本。

**窗口：**

- `"3M"`（自然月）或 `"90D"`（天）表示 [t − 3 个月, t)。自然月按 pandas `DateOffset(months=3)` 推算，3 月 31 日往前 1 个月是 2 月最后一天。
- 一对窗口 `("6M", "3M")` 表示 [t − 6 个月, t − 3 个月)，用于“前一期”，是环比和同比的基础。
- 关键字 `closed="left"`（默认）把 t 当时的事件排除在外。`closed="right"` 改为 (t − k, t]，适用于“截至当日日终”的事件，需要调用者自己确认不会穿越。

**边界情况：**

- 没有任何事件的主体仍得到一行；事件表中有、观察表中没有的主体被忽略。
- 时间列与观察日必须同为无时区或同一时区，否则报错。
- 缺失的时间或主体报错；缺失的数值只从该列的聚合中略去。

## API 提案

两个函数和一个类：

- `window_features` 算时间窗聚合；
- `panel_features` 算还款行为特征；
- `FeatureSpec` 保存两者的定义。

输入是 DataFrame 加列名，与 `fit_spec`、`profile_table` 接收 `frame` 的做法一致，因为一次要聚合多个数值列。

```python
window_features(
    events, observations, *,
    id, time, observation_time,
    windows=("3M", "6M", "12M"),
    aggregations={"amount": ["count", "sum", "mean", "max"]},
    ratios=None, closed="left", prefix="",
)  # -> DataFrame，索引与 observations 相同

panel_features(
    panel, observations, *,
    id, month, status, observation_time,
    windows=("6M", "12M"), bad="30-59", prefix="",
)  # -> DataFrame

spec = FeatureSpec(
    id="customer_id", observation_time="application_date",
    events=[  # 每组是一次 window_features 调用的参数，加上所读事件表的名称 table
        {"time": "time", "windows": ["3M", "12M"], "aggregations": {"amount": ["sum"]}},
        {"table": "cash", "time": "time", "windows": ["3M"],
         "aggregations": {"amount": ["sum"]}, "prefix": "cash_"},
    ],
    ratios=[("cash_amount_sum_3m", "amount_sum_3m")],  # 可跨组
    panel=[  # 每组是一次 panel_features 调用的参数，加上所读面板的名称 table（默认 "panel"）
        {"month": "month", "status": "status", "windows": ["6M", "12M"]},
        {"month": "month", "status": "status", "windows": ["12M"], "bad": "90+", "prefix": "severe_"},
    ],
)
spec.compute(observations, events={"events": events, "cash": cash}, panel=panel)  # 与上面两个函数结果相同
FeatureSpec.from_json(spec.to_json())
```

**列名规则：** `{prefix}{列}_{聚合}_{窗口}`。只有 kivi 自己的部分，即聚合名和窗口，写成小写，窗口写作 `3m`、`90d`、`6m_3m`；调用者给的列名和前缀保持原样，因此 `Amount` 和 `amount` 不会撞名。窗口对必须作为窗口列表中的一项，如 `windows=[("6M", "3M")]`；`windows=("6M", "3M")` 是 `6M` 和 `3M` 两个窗口。

| 定义 | 列名 |
| --- | --- |
| `amount` 近 3 个月的和 | `amount_sum_3m` |
| 近 3 个月的事件数 | `events_3m` |
| `amount` 在 [t − 6 个月, t − 3 个月) 的和 | `amount_sum_6m_3m` |
| 比率 `amount_sum_3m / amount_sum_12m` | `amount_sum_3m_over_12m` |
| 前缀 `cash_` | `cash_amount_sum_3m` |
| 近 12 个月最长连续逾期月数 | `longest_bad_run_12m` |

**输出：** 每个观察一行。列先按窗口、再按列、再按聚合排序，比率列在最后。计数为整数，其余为浮点数。这些列名和顺序一旦发布，就受 1.x 兼容性承诺约束。

**`FeatureSpec`：** 不可变的 dataclass，只存定义，不从数据拟合任何东西，这一点与 `BinningSpec` 不同。所有事件表共用一个 `id` 列。JSON 的结构与构造函数的参数一致：先是 `format: "kivi.FeatureSpec"`、`format_version: 1` 和 `kivi_version`，然后依次是构造函数的参数 `id`、`observation_time`、`events`、`ratios` 和 `panel`。`events` 的每组写出全部键 `table`、`time`、`windows`、`aggregations`（列到聚合列表的对象，按结果中列的顺序）、`closed` 和 `prefix`，元组写成列表，所以 `to_dict()` 的这些键可以直接传给构造函数。读到更新的格式版本时拒绝，JSON 对象中重复的键也报错。`panel` 与 `events` 一样是组的列表，每组写出全部键 `table`、`month`、`status`、`windows`、`bad` 和 `prefix`；`events` 和 `panel` 默认都是空列表 `()`，但至少要有一组；两者都不接受 None，JSON 中的 `null` 也会报错。`panel_features` 与本节一起在 1.3.0 发布，`format_version` 1 由此定稿。

## 语义细节

聚合算子只取一小组定义明确的。窗口为空时，计数和求和为 0，其余为 NaN。

| 聚合 | 含义 | 窗口内没有事件时 |
| --- | --- | --- |
| `count` | 该列非缺失值的个数 | 0 |
| `sum` | 和 | 0 |
| `mean` | 均值 | NaN |
| `min`、`max` | 最小、最大值 | NaN |
| `std` | 样本标准差，与 pandas `std()` 一致 | NaN（少于 2 个值也是 NaN） |
| `nunique` | 不同取值个数，可用于文本列（如商户类型） | 0 |
| `first`、`last` | 窗口内最早、最晚一个事件的值，同一时刻按行序 | NaN |
| `days_since_last` | 观察日到窗口内最后一个事件经过的天数，可含小数；有时区时，它与 `"D"` 窗口按当地日历计的天数可能相差夏令时调整的一小时 | NaN |

`events_{窗口}` 列永远给出，是窗口内的事件行数。

**比率和变化。** `ratios=[("amount_sum_3m", "amount_sum_12m")]` 给出两列之比，分母为 0 时为 NaN。环比就是近期与前一期的比率，如 `amount_sum_3m / amount_sum_6m_3m`；同比用 `("15M", "12M")` 这样的窗口对。差值用 pandas 直接相减即可，不单独提供。

**过滤。** 1.3 不内置条件表达式。比如只统计取现交易时，先用 pandas 筛出这部分事件，再以 `prefix="cash_"` 调用一次；`FeatureSpec` 里可以列出多个这样的组。

**还款行为特征。** `panel_features` 的输入是逐月状态面板，`bad` 是视为“坏”的最低状态：

- `bad_months_{窗口}`：窗口内状态不低于 `bad` 的月数。
- `longest_bad_run_{窗口}`：最长的连续“坏”月数，对应 0.2.x 的连续未还月数。
- `longest_good_run_{窗口}`：最长的连续正常月数，对应连续还款月数。
- `worst_status_{窗口}`：窗口内最差的状态，为有序分类。
- `months_since_bad`：观察日到最近一个“坏”月的月数，从未变坏为 NaN。

只用观察日所在月之前的月末状态，因为当月的月末状态在观察日还不存在。一个主体有多笔贷款时，每月取最差的状态。

实现时补充的决定：

- 每个窗口另有 `months_{窗口}`：窗口内有状态的月数，与 `events_{窗口}` 对应。只统计有状态的月份，贷款开始之前、结清之后或两笔贷款之间没有状态的月份不计入；没有状态的月份会中断连续月数。
- `status` 必须是有序分类，从最好到最差排列，`bad` 必须是其类别之一；字符串列会报错，避免把 `current` 这类标签按字母排序。`bad` 可以是字符串或整数。
- 窗口只能按月，`"90D"` 这类按天的窗口报错；窗口对写成 `windows=[("12M", "6M")]`，与 `window_features` 相同。
- `month` 列可以是月度 Period，也可以是日期时间（取其所在月）；月份和观察时间各按自己的日历取月，因此两边都有时区时必须是同一时区，否则报错（东京的 4 月 1 日在纽约仍是 3 月 31 日，纽约 3 月末的状态那时还不存在）。
- `worst_status_{窗口}` 的 dtype 与 `status` 相同，窗口内没有月份时为缺失；`months_since_bad` 看观察月之前的全部历史，不受窗口限制，从未变坏为 NaN。
- 列名的前缀保持原样，只有 kivi 自己的部分为小写。`worst_status` 不能参与比率。

## 验证、测试与性能

- **参考实现：** 测试里写一个逐个观察循环的暴力实现，在随机数据上逐列比对。覆盖：
  - 同一时刻多个事件；
  - 月末日期；
  - 事件恰好落在窗口边界上；
  - 空窗口；
  - 缺失值。
- **防穿越测试：** 把观察日当天及之后的事件改成极端值，特征必须不变。
- **与 pandas 对照：** 对按天的窗口，在观察日恰好等于事件时间的特例下，与 `groupby().rolling("90D", closed="left")` 对照。
- **`FeatureSpec`：** JSON 往返后相等；另存一份 golden 文件，与 `BinningSpec` 的做法相同。
- **性能：**
  - 按（主体，时间）排序后，用 `searchsorted` 找每个观察的窗口起止位置；
  - `count`、`sum`、`mean` 用累计和相减；
  - `min`、`max` 用分段 reduce；
  - `nunique` 按组处理。
  
  目标：100 万事件、10 万个观察、3 个窗口、5 个聚合在 10 秒内完成。

## 文档与实施计划

1. 本设计稿审定：2026-11-23 已确认，下一节的 9 项全部采用推荐。
2. 并行两个会话：
    - **A：** `kivi.features` 的 `window_features` 和 `FeatureSpec`，包括新模块、测试、指南 `features.rst` 和 API 页。
    - **B：** `kivi.datasets.make_transactions`，为 `make_applications` 的申请人模拟申请日之前的交易流水。流水与隐藏的风险因子相关，供指南演示 IV。
3. A 完成后接着做 `panel_features`。它与 A 在同一模块，所以顺序进行。
4. 集成 PR：
    - README、各级索引和兼容性页；
    - 迁移指南：写清 `FeatureTools`、`FeatureToolsV2`、`index/` 与 `kivi.features` 的对应；
    - CLAUDE.md。
5. 发布 1.3.0，需要你的确认。

## 决定

2026-11-23，以下 9 项全部采用推荐方案。

| # | 问题 | 决定 | 未采用的选项 |
| --- | --- | --- | --- |
| 1 | 输入形式 | DataFrame 加列名 | 一维数组，与 `vintage_table` 一致，但多列聚合要传很多参数 |
| 2 | 窗口边界 | 默认 [t − k, t)，可用 `closed="right"` | 默认 (t − k, t] |
| 3 | 窗口单位 | 同时支持 `"3M"`（自然月）和 `"90D"`（天） | 只支持天 |
| 4 | 空窗口取值 | `count`、`sum`、`nunique` 为 0，其余 NaN | 全部 NaN |
| 5 | 比率是否进 `FeatureSpec` | 进，保证开发与生产一致 | 用户自己用 pandas 算 |
| 6 | 条件过滤 | 不内置，先用 pandas 筛选再加 `prefix` | 内置 `where` 表达式 |
| 7 | 还款行为特征 | 1.3 做，作为第二个 PR | 推迟到以后 |
| 8 | 模拟交易数据 `make_transactions` | 新增，供指南和测试使用 | 只用手工小例子 |
| 9 | 群体内百分位排名 | 不在 1.3 | 1.3 一并做 |
