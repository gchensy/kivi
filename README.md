# kivi

用 pandas 做二分类建模中的变量分箱、证据权重（WOE）、信息值（IV）和群体稳定性指标（PSI）分析。支持 Python 3.10 及以上版本。

Kivi provides binning, weight of evidence (WOE), information value (IV) and population stability (PSI) for binary modeling with pandas. Requires Python 3.10 or newer.

**从源码安装 · Install from a local checkout**

```bash
python -m pip install -e .
```

```python
import pandas as pd
from kivi.datasets import Dataset
from kivi.binning import bin_edges
from kivi.woe import woe_table
from kivi.evaluate import psi_table

data = Dataset.bank_data()
reference, current = data.iloc[:3000], data.iloc[3000:]
edges = bin_edges(reference["age"], bins=5)
before = pd.cut(reference["age"], edges)
after = pd.cut(current["age"], edges)
evidence = woe_table(before, reference["target"])
stability = psi_table(before, after)
print(evidence[["event", "non_event", "woe", "iv"]])
print(stability["psi"].sum())
```

**文档 · Documentation**: <https://gchensy.github.io/kivi/>

文档包含快速上手、指标口径、API 参考和旧版迁移说明。Both the English and Chinese manuals include a quickstart, metric definitions, API reference and migration guide.

**构建与测试 · Build and test**

```bash
python -m pip install -e '.[docs,test]'
python -m pytest
python docs/build.py
```

构建后打开 `docs/_build/html/index.html`。Open `docs/_build/html/index.html` to choose a language. The code is MIT licensed.

源代码采用 MIT 许可证。内置数据的作者、用途和授权见 [数据说明](DATA-LICENSE.txt)。
