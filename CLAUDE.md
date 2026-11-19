# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
python -m pip install -e '.[test,docs]'    # required first: kivi.__version__ and docs/conf.py read installed package metadata
python -m pytest                            # unit tests
python -m pytest tests/test_core.py::test_woe_matches_hand_calculation   # single test
python -m sphinx -W -b doctest docs docs/_build/doctest   # runs the >>> examples in docs/*.rst (pytest does not collect them)
python docs/build.py                        # en + zh_CN HTML into docs/_build/html; uses -W, so any Sphinx warning fails the build
```

No linter or formatter is configured. An editable install resolves to the checkout it was installed from; when working in a git worktree, prefix commands with `PYTHONPATH=src` (or reinstall) so they exercise the worktree's code.

## Architecture

`src/kivi` (src layout, hatchling) is the 0.3 rebuild of the legacy 0.2.x library, reduced to a small NumPy/pandas core. There are no top-level re-exports: the public API is imported from submodules, and `docs/api.rst` lists exactly the four public entry points.

Fitting edges and grouping are deliberately separate steps:

1. `kivi.binning.bin_edges(values, bins, method)` fits edges on the reference sample only. Interior duplicate edges are collapsed and the endpoints are always `-inf`/`+inf`, so future out-of-range values still land in a bin (a constant reference yields one bin).
2. The caller applies `pd.cut(series, edges)` with the same edges to every sample. Categorical variables skip steps 1–2 and are passed as-is.
3. `kivi.woe.woe_table(groups, target)` and `kivi.evaluate.psi_table(reference, current)` take already-grouped series and return one row per group. Totals are the caller's `.sum()` of the `iv` / `psi` column.

Both metric functions share `_counts.py`, which defines the counting and smoothing semantics — change behavior there, not per metric:

- `group_counts` uses `pd.factorize(sort=False, use_na_sentinel=False)`: missing values form their own group, groups appear in first-observed order, and unused categorical levels are dropped. The result index is named `bin`.
- `shares` replaces zero cells with `smoothing` (default 0.5) and then normalizes each column independently. Count columns in the output stay raw; only the `*_share` columns reflect smoothing.

Output columns: `woe_table` → `non_event, event, non_event_share, event_share, woe, iv`; `psi_table` → `reference, current, reference_share, current_share, psi`. `woe_table` requires `groups` and `target` to have identical indices and both classes present.

`kivi.datasets.Dataset.bank_data()` loads the bundled `;`-separated `bank.csv` (UCI Bank Marketing, CC BY 4.0, see `DATA-LICENSE.txt`) via `importlib.resources` and maps `y` yes/no to `target` 1/0. Tests pin its shape (4521×17) and class counts.

## Bilingual documentation

- Sphinx with pydata-sphinx-theme and numpydoc; docstrings are numpydoc-formatted and rendered through autodoc in `docs/api.rst`.
- The Chinese edition comes from one gettext catalog, `docs/locale/zh_CN/LC_MESSAGES/kivi.po` (`gettext_compact = "kivi"`); `.mo` files are gitignored and compiled during the build. Each msgid is the exact English string, **including public docstring paragraphs**, so editing English text in `docs/*.rst` or a docstring requires updating the matching entry — otherwise that string silently falls back to English in the zh build. `python -m sphinx -b gettext docs docs/_build/gettext` writes `kivi.pot` to compare against.
- `docs/build.py` builds `en/` and `zh/` as separate outputs (separate search indices) plus a root language-picker `index.html`. `_templates/language-switch.html` links to the same page name in the other language, so both editions must keep identical page names.
- `README.md` is bilingual (Chinese and English); keep both languages when editing it.

## Scope and design constraints

From `docs/development.rst` and `docs/migration.rst`:

- The core depends only on NumPy and pandas. Infrastructure-specific code (Spark, SQL, etc.) belongs in optional integrations and must not be imported at module load.
- Functions must not mutate caller inputs; anything random takes a local seed instead of touching NumPy's global state.
- Weighted/supervised binning, scorecards, Spark, SQL, AHP, feature selection and model wrappers exist only in git history before commit 740ea6c. When porting one, validate against fixed reference samples first.
- Tests assert raw counts, normalized shares and hand-calculated metric values (the legacy tests only printed tables) — follow that pattern.
