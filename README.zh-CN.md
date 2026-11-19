<div align="center">

<img src="https://raw.githubusercontent.com/gchensy/kivi/master/docs/_static/kivi-mark.svg" width="88" alt="kivi logo">

# kivi

**基于 pandas 的分箱、WOE、IV 与 PSI**

拟合一次分箱边界，看清每一箱对二分类目标的区分能力，<br>
并持续监测样本分布的漂移——结果都是普通的 pandas DataFrame。

[![Documentation](https://github.com/gchensy/kivi/actions/workflows/docs.yml/badge.svg)](https://github.com/gchensy/kivi/actions/workflows/docs.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![pandas 2.0+](https://img.shields.io/badge/pandas-2.0%2B-150458?logo=pandas&logoColor=white)](https://pandas.pydata.org/)
[![文档：中文 | English](https://img.shields.io/badge/%E6%96%87%E6%A1%A3-%E4%B8%AD%E6%96%87%20%7C%20English-2f5bd3)](https://gchensy.github.io/kivi/)
[![License: MIT](https://img.shields.io/badge/license-MIT-0b7a70)](https://github.com/gchensy/kivi/blob/master/LICENSE)
[![GitHub stars](https://img.shields.io/github/stars/gchensy/kivi?style=social)](https://github.com/gchensy/kivi)

[English](https://github.com/gchensy/kivi/blob/master/README.md) · **简体中文**

[使用文档](https://gchensy.github.io/kivi/zh/) ·
[快速上手](https://gchensy.github.io/kivi/zh/getting_started/quickstart.html) ·
[用户指南](https://gchensy.github.io/kivi/zh/user_guide/index.html) ·
[API 参考](https://gchensy.github.io/kivi/zh/api.html) ·
[版本说明](https://gchensy.github.io/kivi/zh/release/index.html)

</div>

---

## ✨ 特性

- 📐 **一次拟合，处处复用。** `bin_edges` 在基准样本上学习等频或等距边界，两端延伸至无穷，之后的任何取值都能落入某个箱。
- 🔍 **结果透明。** `woe_table` 和 `psi_table` 返回每箱的原始计数、平滑后的占比和单箱贡献，而不只是一个总数。
- 🧩 **不悄悄丢数据。** 缺失值、只在一个样本中出现的类别，都会单独成组。
- 🧊 **平滑口径清楚。** 空单元格先按 `smoothing`（默认 0.5）替换再归一化，结果中的计数始终是实际观测数。
- 🐼 **只依赖 pandas 和 NumPy。** 核心不依赖 Spark 和数据库驱动，也不修改全局随机状态。
- 🌏 **中英双语文档，示例全部经过测试。** 两种语言文档中的每个示例都会在 CI 中运行。

## 🧭 工作流程

<p align="center">
  <img src="https://raw.githubusercontent.com/gchensy/kivi/master/docs/_static/kivi-workflow.zh-CN.svg" width="820" alt="用 bin_edges 拟合一次边界，用 pandas.cut 对每个样本分箱，再用 woe_table 和 psi_table 计算指标">
</p>

## 📦 安装

```bash
pip install "git+https://github.com/gchensy/kivi"
```

需要 Python 3.10+、NumPy 1.24+ 和 pandas 2.0+。

> **注意：** 目前 `pip install kivi` 安装的是 0.2.8，其 API 与当前版本不同。升级为 0.2.x 编写的代码前，请先阅读[迁移指南](https://gchensy.github.io/kivi/zh/release/migration.html)。

## 🚀 快速上手

```python
import pandas as pd
from kivi.binning import bin_edges
from kivi.datasets import Dataset
from kivi.evaluate import psi_table
from kivi.woe import woe_table

data = Dataset.bank_data()                      # 4,521 位银行营销客户
reference, current = data.iloc[:3000], data.iloc[3000:]

edges = bin_edges(reference["age"], bins=5)     # 1. 只在基准样本上拟合
before = pd.cut(reference["age"], edges)        # 2. 每个样本复用同一组边界
after = pd.cut(current["age"], edges)

evidence = woe_table(before, reference["target"]).sort_index()   # 3. 计算指标
stability = psi_table(before, after).sort_index()

print(evidence[["event", "non_event", "woe", "iv"]].round(3))
print(f"IV = {evidence['iv'].sum():.3f}, PSI = {stability['psi'].sum():.4f}")
```

```text
              event  non_event    woe     iv
bin
(-inf, 32.0]     90        612  0.114  0.003
(32.0, 36.0]     55        475 -0.125  0.003
(36.0, 42.0]     58        529 -0.180  0.006
(42.0, 51.0]     71        569 -0.050  0.001
(51.0, inf]      74        467  0.189  0.007
IV = 0.019, PSI = 0.0024
```

类别变量无需边界，直接传入：

```python
jobs = woe_table(reference["job"], reference["target"])
print(f"IV(job) = {jobs['iv'].sum():.3f}")      # IV(job) = 0.137
```

## 🧰 API 一览

| 函数 | 用途 |
| --- | --- |
| [`kivi.binning.bin_edges`](https://gchensy.github.io/kivi/zh/api.html#kivi.binning.bin_edges) | 在基准样本上拟合等频或等距边界，两端为无穷 |
| [`kivi.woe.woe_table`](https://gchensy.github.io/kivi/zh/api.html#kivi.woe.woe_table) | 每组的计数、占比、WOE 与 IV |
| [`kivi.evaluate.psi_table`](https://gchensy.github.io/kivi/zh/api.html#kivi.evaluate.psi_table) | 两个样本逐组的 PSI 贡献 |
| [`kivi.datasets.Dataset.bank_data`](https://gchensy.github.io/kivi/zh/api.html#kivi.datasets.Dataset.bank_data) | 随包的银行营销示例数据 |

## 📚 文档

| | 中文 | English |
| --- | --- | --- |
| 快速上手 | [阅读](https://gchensy.github.io/kivi/zh/getting_started/quickstart.html) | [Read](https://gchensy.github.io/kivi/en/getting_started/quickstart.html) |
| 分箱 | [阅读](https://gchensy.github.io/kivi/zh/user_guide/binning.html) | [Read](https://gchensy.github.io/kivi/en/user_guide/binning.html) |
| WOE 与 IV | [阅读](https://gchensy.github.io/kivi/zh/user_guide/woe_iv.html) | [Read](https://gchensy.github.io/kivi/en/user_guide/woe_iv.html) |
| PSI | [阅读](https://gchensy.github.io/kivi/zh/user_guide/psi.html) | [Read](https://gchensy.github.io/kivi/en/user_guide/psi.html) |
| 从 0.2.x 迁移 | [阅读](https://gchensy.github.io/kivi/zh/release/migration.html) | [Read](https://gchensy.github.io/kivi/en/release/migration.html) |

## 🗺️ 路线图

- ✅ 基于 pandas 核心的等频与等距分箱、WOE、IV 和 PSI
- ✅ 中英文文档，示例经过测试
- 🔜 评分卡流程，在固定参考样本上完成结果核对
- 🔜 加权分箱与监督分箱
- 🔜 Spark 与 SQL 作为可选集成

## 🤝 参与贡献

```bash
git clone https://github.com/gchensy/kivi.git
cd kivi
python -m pip install -e '.[test,docs]'
python -m pytest                                          # 单元测试
python -m sphinx -W -b doctest docs docs/_build/doctest   # 文档示例
python docs/build.py                                      # 中英文 HTML 文档
```

修改英文文档或 docstring 后，运行 `python docs/build.py --update-po`，并补全 `docs/locale/zh_CN/LC_MESSAGES/kivi.po` 中的新条目。详见[开发指南](https://gchensy.github.io/kivi/zh/development/index.html)。

## 📄 许可

源代码采用 [MIT 许可证](https://github.com/gchensy/kivi/blob/master/LICENSE)。内置的 Bank Marketing 数据由 S. Moro、P. Rita 和 P. Cortez（2014）发布，按 CC BY 4.0 授权，详见 [DATA-LICENSE.txt](https://github.com/gchensy/kivi/blob/master/DATA-LICENSE.txt)。

<div align="center">

<sub>如果 kivi 对你的模型有帮助，欢迎点一个 ⭐，让更多人看到它。</sub>

</div>
