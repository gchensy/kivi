# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
python -m pip install -e '.[test,docs]'    # required first: kivi.__version__ and docs/conf.py read installed package metadata
python -m pytest                            # unit tests
python -m pytest tests/test_core.py::test_woe_matches_hand_calculation   # single test
python -m sphinx -W -b doctest docs docs/_build/doctest   # runs every >>> example in docs/**/*.rst and in docstrings (pytest does not collect them)
python docs/build.py                        # checks the zh_CN catalog, then builds en + zh HTML into docs/_build/html with -W
python docs/build.py --update-po            # after editing English text: merge new/changed strings into kivi.po
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

- Sphinx with pydata-sphinx-theme, numpydoc, sphinx-design (cards, tabs, buttons), sphinx-copybutton and MathJax. Pages are grouped by section folder (`getting_started/`, `user_guide/`, `release/`, `development/`) plus top-level `index.rst` and `api.rst`; the hidden toctree in `index.rst` defines the navbar. Brand colors, fonts and the home-page layout live in `docs/_static/kivi.css`; navbar extras are `_templates/version-badge.html` and `_templates/language-switch.html`.
- Examples are doctests with real output (`NORMALIZE_WHITESPACE` is on). If a change alters a printed DataFrame, regenerate the output by running the code rather than editing numbers by hand.
- The Chinese edition comes from one gettext catalog, `docs/locale/zh_CN/LC_MESSAGES/kivi.po` (`gettext_compact = "kivi"`; `.mo` files are gitignored). Each msgid is the exact English string, **including public docstring paragraphs, table cells and math blocks**. `docs/build.py` regenerates `kivi.pot` and fails on untranslated, fuzzy or stale entries, so after any English edit run `--update-po`, fill the empty/fuzzy `msgstr`s and remove the `#, fuzzy` flags. In translations keep a space (or CJK punctuation) around RST inline markup such as ``` ``code`` ``` and `` :doc:`x` ``, or docutils fails to parse it.
- sphinx-design button labels are not extracted for translation; the home-page buttons therefore have no explicit text and show the target page's (translated) title.
- `docs/build.py` deletes and rebuilds `docs/_build/html`: `en/` and `zh/` are separate Sphinx builds (separate search indices), `docs/landing.html` is copied to the root as the language chooser, and `.nojekyll` is added for GitHub Pages. The language switch links to the same page name in the other edition, so both editions must keep identical page names.
- `conf.py` replaces Sphinx's Chinese search language with a subclass because Sphinx 9.0 emits `window.Stemmer = ChineseStemmer` (undefined), which breaks search on the zh pages. Keep the workaround until Sphinx fixes it.
- `.github/workflows/docs.yml` runs pytest, the doctests and `docs/build.py` on every push and pull request, and on master force-pushes `docs/_build/html` to the `gh-pages` branch (served at https://gchensy.github.io/kivi/), replacing whatever was there.
- `README.md` (English, also the PyPI long description) and `README.zh-CN.md` (Chinese) mirror each other; keep them in sync. Because PyPI cannot resolve relative paths, every image and link in them is an absolute `raw.githubusercontent.com/.../master/` or `github.com/.../blob/master/` URL. The workflow diagrams are `docs/_static/kivi-workflow.svg` and `kivi-workflow.zh-CN.svg`, with dark-mode colors in an embedded `prefers-color-scheme` style. The quickstart output in both READMEs is pasted from a real run; regenerate it if the example or the data changes.

## Releasing

`.github/workflows/release.yml` publishes to PyPI with trusted publishing (GitHub environment `pypi`, no stored token) when a GitHub release is published. The release tag must be `v` plus the exact `version` in `pyproject.toml` (e.g. `v0.3.0`) or the build job fails before anything is uploaded; it also runs pytest and `twine check --strict` (which renders `README.md`). To release: bump `version`, date the entry in `docs/release/index.rst` (and translate it), merge to master, then publish a GitHub release tagged `v<version>`.

## Scope and design constraints

From `docs/development/legacy_review.rst` and `docs/release/migration.rst`:

- The core depends only on NumPy and pandas. Infrastructure-specific code (Spark, SQL, etc.) belongs in optional integrations and must not be imported at module load.
- Functions must not mutate caller inputs; anything random takes a local seed instead of touching NumPy's global state.
- Weighted/supervised binning, scorecards, Spark, SQL, AHP, feature selection and model wrappers exist only in git history before commit 740ea6c. When porting one, validate against fixed reference samples first.
- Tests assert raw counts, normalized shares and hand-calculated metric values (the legacy tests only printed tables) — follow that pattern.
