# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
python -m pip install -e '.[test,docs,lint]'   # required first: docs/conf.py reads installed package metadata
pre-commit install                          # once per clone: run ruff on every commit
ruff check . && ruff format --check .       # what CI's build job runs before the tests
python -m pytest                            # unit tests
python -m pytest tests/test_core.py::test_woe_matches_hand_calculation   # single test
python -m sphinx -W -b doctest docs docs/_build/doctest   # runs every >>> example in docs/**/*.rst and in docstrings (pytest does not collect them)
python docs/build.py                        # checks the zh_CN catalog, then builds en + zh HTML into docs/_build/html with -W
python docs/build.py --update-po            # after editing English text: merge new/changed strings into kivi.po
python docs/build.py --merge-po             # after a merge/rebase stops on kivi.po: keep both sides' translations, then update
python docs/build.py --release 0.4.0        # at release time: move docs/release/unreleased/* into a section dated today
```

ruff lints and formats all Python files (`[tool.ruff]` in `pyproject.toml`: line length 100, rules E/F/W/I/B/UP/NPY/D with the numpy docstring convention; tests and docs scripts are exempt from the D rules). The version is pinned in the `lint` extra and as `rev` in `.pre-commit-config.yaml`; bump both together. Markdown is excluded because the README code blocks align their comments by hand. pytest runs with `filterwarnings = error`, so assert expected warnings with `pytest.warns`. `numpydoc_validation_checks = {"all"}` turns any missing docstring section (including See Also) into a `-W` build failure. `conf.py` also turns warnings raised from doctest code or from `docs/_plotting.py` into errors, so an example that demonstrates a warning must catch it with `warnings.catch_warnings(record=True)`. An editable install resolves to the checkout it was installed from; when working in a git worktree, prefix commands with `PYTHONPATH=src` (or reinstall) so they exercise the worktree's code.

## Architecture

`src/kivi` (src layout, hatchling) is the 0.3 rebuild of the legacy 0.2.x library, reduced to a small NumPy/pandas core. There are no top-level re-exports: the public API is imported from submodules, and each public module has its own page in `docs/api/<module>.rst` listing every entry point.

Fitting edges and grouping are deliberately separate steps:

1. `kivi.binning.bin_edges(values, bins, *, method, special)` fits edges on the reference sample only. Interior duplicate edges are collapsed and the endpoints are always `-inf`/`+inf`, so future out-of-range values still land in a bin (a constant reference yields one bin). The supervised `chimerge_edges`, `tree_edges` and `monotonic_edges(values, target, ...)` in the same module return edges of the same form. ChiMerge and monotonic binning start from `initial_bins` quantile bins (one bin per distinct value when there are no more distinct values than that) and merge adjacent pairs; the ChiMerge p-value uses `math.erfc` (one degree of freedom), so there is no scipy dependency. `monotonic_edges` checks the trend with `_counts.shares`, so `woe_table` with the same `smoothing` sees the same monotonic WOE. `tree_edges` grows best first and its interior edges are observed values; it matches scikit-learn's `DecisionTreeClassifier(max_leaf_nodes=max_bins, min_samples_leaf=min_bin_share)` splits, which is a useful cross-check but not a dependency.
2. The caller applies the same edges to every sample with `kivi.binning.cut(series, edges, special=...)`, or `pd.cut` when there are no special values. `special` codes (e.g. bank `pdays == -1`) are left out of the fit in step 1 (the supervised functions drop those rows together with their labels in `_labeled`), and `cut` gives each its own category ahead of the intervals; without `special` it returns exactly `pd.cut`'s result. Categorical variables skip steps 1–2 and are passed as-is.
3. `kivi.woe.woe_table(groups, target)` and `kivi.stability.psi_table(reference, current)` take already-grouped series and return one row per group. Totals are the caller's `.sum()` of the `iv` / `psi` column.
4. `kivi.woe.woe_transform(groups, table, unseen="raise")` maps groups back to the fitted WOE (the scorecard encoding). It factorizes the groups once, looks each distinct label up in the table by equality (intervals from the same edges match), and gives missing values the table's missing-group WOE. A label absent from the table raises unless `unseen` is a number, so new categories and empty development intervals never turn into silent NaN.

Deprecations warn with `FutureWarning` for one minor version before removal. `_deprecation.positional_deprecated("method", removed_in="0.6.0")` keeps the documented keyword-only signature of `bin_edges` while still accepting `method` by position, and `kivi.evaluate` is a module-level `__getattr__` alias of `kivi.stability`. New code and docs must not use either pattern, because pytest and the doctests treat warnings as errors.

Public functions accept one-dimensional array_like input through `_validation.as_series`, which leaves Series untouched. `woe_table` and the supervised binning functions require identical indices only when both inputs are Series; otherwise they match by position. Error messages name the argument and the value received, and tests check them with `pytest.raises(..., match=)`.

All metric functions share `_counts.py`, which defines the counting and smoothing semantics. Change behavior there, not per metric:

- `group_counts` uses `pd.factorize(use_na_sentinel=False)`: missing values form their own group and unused categorical levels are dropped. Categorical groups are factorized with `sort=True` (category order, missing last); other groups with `sort=False` (first-observed order). The result index is named `bin`.
- Counting uses `np.bincount` on `2 * code + side`.
- `shares` replaces zero cells with `smoothing` (default 0.5) and then normalizes each column independently. Count columns in the output stay raw; only the `*_share` columns reflect smoothing.
- `divergence(counts, smoothing)` takes an `(n, 2)` count array and returns the shares, `ln(b / a)` and `(b - a) ln(b / a)` per row. With (non-event, event) columns these are WOE and IV, and with (reference, current) columns the PSI terms. `woe_table`, `psi_table`, the count-based `kivi.woe.woe` / `kivi.woe.iv` / `kivi.stability.psi` and the supervised binning helpers all go through it. The public count functions validate their input with `_validation.as_count_columns`.

Output columns: `woe_table` → `non_event, event, total, non_event_share, event_share, total_share, event_rate, woe, iv` (only the two class shares are smoothed; `total_share` and `event_rate` use raw counts); `psi_table` → `reference, current, reference_share, current_share, psi`. `woe_table` requires `groups` and `target` to have identical indices and both classes present.

`kivi.datasets.Dataset.bank_data()` loads the bundled `;`-separated `bank.csv` (UCI Bank Marketing, CC BY 4.0, see `DATA-LICENSE.txt`) via `importlib.resources` and maps `y` yes/no to `target` 1/0. Tests pin its shape (4521×17) and class counts.

## Bilingual documentation

- Sphinx with pydata-sphinx-theme, numpydoc, sphinx-design (cards, tabs, buttons), sphinx-copybutton and MathJax. Pages are grouped by section folder (`getting_started/`, `user_guide/`, `release/`, `development/`) plus top-level `index.rst` and `api.rst`; the hidden toctree in `index.rst` defines the navbar. `api.rst` and `user_guide/index.rst` end their toctrees with a glob (`api/*`, `*`), so a new `docs/api/<module>.rst` or `docs/user_guide/<topic>.rst` appears without editing them. Each glob must always match a page, or Sphinx warns and `-W` fails, so `api/datasets` and `datasets` are deliberately left out of the explicit lists. Brand colors, fonts and the home-page layout live in `docs/_static/kivi.css`; navbar extras are `_templates/version-badge.html` and `_templates/language-switch.html`.
- Examples are doctests with real output (`NORMALIZE_WHITESPACE` is on). If a change alters a printed DataFrame, regenerate the output by running the code rather than editing numbers by hand.
- The docs are positioned for credit risk and financial modeling; `user_guide/risk_workflow.rst` is the end-to-end example (IV screening, binning, WOE encoding, PSI report). Charts are drawn at build time by matplotlib's `.. plot::` directive calling `figure_*` recipes in `docs/_plotting.py` (imported as `kp` via `plot_pre_code`). Each recipe recomputes the example shown in the doctest just above it, so change both together. In-chart text uses only kivi column and parameter names (`woe`, `event_rate`, `reference_share`, `quantile`…) and data values, so one image serves both languages; captions and alt text are translated through the catalog. Keep it that way: CJK text inside a chart would need CJK fonts on CI.
- The Chinese edition comes from one gettext catalog, `docs/locale/zh_CN/LC_MESSAGES/kivi.po` (`gettext_compact = "kivi"`; `.mo` files are gitignored). Each msgid is the exact English string, **including public docstring paragraphs, table cells and math blocks**. `docs/build.py` regenerates `kivi.pot` and fails on untranslated, fuzzy or stale entries, so after any English edit run `--update-po`, fill the empty/fuzzy `msgstr`s and remove the `#, fuzzy` flags. `--update-po` keeps the header's `POT-Creation-Date`, so pull requests only conflict where they change the same strings or neighbouring entries; resolve such conflicts with `--merge-po`, never by hand. In translations keep a space (or CJK punctuation) around RST inline markup such as ``` ``code`` ``` and `` :doc:`x` ``, or docutils fails to parse it.
- sphinx-design button labels are not extracted for translation; the home-page buttons therefore have no explicit text and show the target page's (translated) title.
- `docs/build.py` deletes and rebuilds `docs/_build/html`: `en/` and `zh/` are separate Sphinx builds (separate search indices), `docs/landing.html` is copied to the root as the language chooser, and `.nojekyll` is added for GitHub Pages. The language switch links to the same page name in the other edition, so both editions must keep identical page names.
- `conf.py` replaces Sphinx's Chinese search language with a subclass because Sphinx 9.0 emits `window.Stemmer = ChineseStemmer` (undefined), which breaks search on the zh pages. Keep the workaround until Sphinx fixes it.
- `.github/workflows/docs.yml` runs ruff, pytest, the doctests and `docs/build.py` in the required `build` job on every push and pull request (keep that job name and do not turn it into a matrix, or the required check never reports). A separate `minimum-versions` job runs pytest on Python 3.10 with NumPy 1.24.4 and pandas 2.0.3, the oldest versions `pyproject.toml` allows. On master, the `build` job also force-pushes `docs/_build/html` to the `gh-pages` branch (served at https://gchensy.github.io/kivi/), replacing whatever was there.
- `README.md` (English, also the PyPI long description) and `README.zh-CN.md` (Chinese) mirror each other; keep them in sync. Because PyPI cannot resolve relative paths, every image and link in them is an absolute `raw.githubusercontent.com/.../master/` or `github.com/.../blob/master/` URL. The workflow diagrams are `docs/_static/kivi-workflow.svg` and `kivi-workflow.zh-CN.svg`, with dark-mode colors in an embedded `prefers-color-scheme` style. The quickstart output in both READMEs is pasted from a real run; regenerate it if the example or the data changes. The gallery images `docs/_static/readme-*.svg` are committed outputs of `python docs/_plotting.py` (deterministic thanks to `svg.hashsalt`); rerun it when a gallery recipe or its data changes.

## Releasing

`.github/workflows/release.yml` runs when a `v*` tag is pushed. The build job fails before anything is uploaded unless the tag is `v` plus the exact `version` in `pyproject.toml` and the tagged commit is on master; it also runs pytest and `twine check --strict` (which renders `README.md`). The publish job uploads to PyPI with trusted publishing through the `pypi` environment (no stored token), and only after that succeeds does a third job create the GitHub release with `gh release create --verify-tag --generate-notes` and the dist files attached (`a`/`b`/`rc`/`dev` versions become pre-releases). Do not create the GitHub release by hand first: the job would fail on the existing release.

Protection lives in repository settings, not in the code: the `pypi` environment only accepts deployments from `v*` tags and has no required reviewers, so a pushed tag is uploaded without further approval; a tag ruleset ("Release tags") lets only admins create, move or delete `v*` tags; PyPI's trusted publisher is bound to `gchensy/kivi` + `release.yml` + environment `pypi`, so renaming the workflow file or environment breaks publishing until PyPI is updated.

To release: bump `version`, run `python docs/build.py --release <version>` (it writes the dated section from the fragments and deletes them) and translate the new heading, merge to master, then `git tag v<version>` on the merged commit, and `git push origin v<version>`. The push publishes to PyPI; there is no approval step to stop it.

## Development workflow

Several Claude sessions often work on this repository at once. These rules keep their pull requests mergeable:

- **Scope first.** The session that splits up the work gives every parallel session a brief naming the files it owns. Two in-flight pull requests must not edit the same file; work that would is done one after the other, and a session that finds it needs a file another open PR changes says so instead of editing it. Usual hotspots: `src/kivi/binning.py`, `src/kivi/_counts.py`, `docs/conf.py`, `docs/user_guide/binning.rst`, `CLAUDE.md`.
- **A feature session only adds its own files**: `src/kivi/<module>.py`, `tests/test_<module>.py`, `docs/user_guide/<topic>.rst`, `docs/api/<module>.rst` and a release-note fragment. The globs pick up the new pages. `kivi.po` is the one shared file every PR may change; resolve its conflicts with `--merge-po`. Feature PRs must not add dependencies.
- **Shared entry points change only in an integration PR** that the coordinating session opens after a wave of feature PRs has merged: the README tables and roadmap (both languages), the cards in `docs/index.rst` and `docs/user_guide/index.rst`, `docs/user_guide/risk_workflow.rst` and this file.
- **Branch from the latest `origin/master`** (fetch first), one topic per pull request. Stack a pull request on an unmerged branch only when the work depends on it, and say so in its description.
- **Release notes go in fragments**: add `docs/release/unreleased/<topic>.<category>.rst` (category `new`, `changed`, `fixed` or `removed`) with one bullet item per user-facing change. Never edit the Unreleased section of `docs/release/index.rst`; the `release-fragments` directive in `docs/_fragments.py` renders the fragments there. A fragment is translated like any other text.
- **master is protected** by the repository ruleset "master": no direct or force pushes and no deletion; a change lands through a pull request whose `build` check passed on a branch that is up to date with master, so every PR is tested on top of everything merged before it. Merged branches are deleted automatically.
- **Merging is done on request.** A session merges only pull requests the user asked it to merge, in dependency order: bring the branch up to date by merging `origin/master` into it (not rebasing; PRs with Auto-fix must not be force-pushed), resolve a `kivi.po` conflict with `--merge-po`, wait for `build`, merge with `gh pr merge --merge`, then tell the sessions whose open PRs touch the same files to merge master.
- **Decisions that stay with the user**: priorities and scope, changes to the public API, merges that were not delegated, and releases (pushing a `v*` tag, which publishes to PyPI at once and cannot be undone).

## Scope and design constraints

From `docs/development/legacy_review.rst` and `docs/release/migration.rst`:

- The core depends only on NumPy and pandas. Infrastructure-specific code (Spark, SQL, etc.) belongs in optional integrations and must not be imported at module load.
- Functions must not mutate caller inputs; anything random takes a local seed instead of touching NumPy's global state.
- Weighted binning, scorecards, Spark, SQL, AHP, feature selection and model wrappers exist only in git history before commit 740ea6c. When porting one, validate against fixed reference samples first. Supervised binning was re-implemented rather than ported; `docs/release/migration.rst` lists how it differs from the legacy `Chi2Bins` and `TreeBins`.
- Tests assert raw counts, normalized shares and hand-calculated metric values (the legacy tests only printed tables) — follow that pattern.
