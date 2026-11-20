import doctest
import sys
from importlib.metadata import version as distribution_version
from pathlib import Path

import sphinx.search
from sphinx.search.zh import SearchChinese


class ChineseSearch(SearchChinese):
    # Sphinx 9.0 ships english-stemmer.js for Chinese but assigns
    # `window.Stemmer = ChineseStemmer`, which is undefined and stops the search page.
    language_name = "English"


if SearchChinese.js_stemmer_rawcode == "english-stemmer.js":
    sphinx.search.languages["zh"] = ChineseSearch

# Lets the plot directives import docs/_plotting.py.
sys.path.insert(0, str(Path(__file__).resolve().parent))

project = "kivi"
author = "Chensy"
copyright = "2024, Chensy"
release = distribution_version("kivi")
version = release

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.doctest",
    "sphinx.ext.mathjax",
    "numpydoc",
    "sphinx_copybutton",
    "sphinx_design",
    "matplotlib.sphinxext.plot_directive",
]
numpydoc_show_class_members = False
# The numpydoc Parameters section documents the types; npt.ArrayLike would expand to a long union.
autodoc_typehints = "none"
numpydoc_validation_checks = {"all"}
doctest_default_flags = (
    doctest.ELLIPSIS
    | doctest.IGNORE_EXCEPTION_DETAIL
    | doctest.DONT_ACCEPT_TRUE_FOR_1
    | doctest.NORMALIZE_WHITESPACE
)
copybutton_prompt_text = r">>> |\.\.\. |\$ "
copybutton_prompt_is_regexp = True

# Charts are drawn at build time from the bundled data by docs/_plotting.py.
plot_pre_code = "import _plotting as kp\nkp.apply_style()\n"
plot_include_source = False
plot_html_show_source_link = False
plot_html_show_formats = False
plot_formats = ["svg"]

language = "en"
locale_dirs = ["locale/"]
gettext_compact = "kivi"
exclude_patterns = ["_build", "_templates", "landing.html"]
templates_path = ["_templates"]

html_theme = "pydata_sphinx_theme"
html_title = "kivi"
html_static_path = ["_static"]
html_css_files = ["kivi.css"]
html_favicon = "_static/kivi-mark.svg"
html_show_sourcelink = False
html_copy_source = False
html_sidebars = {"**": ["sidebar-nav-bs"], "index": [], "api": []}
html_theme_options = {
    "logo": {
        "text": "kivi",
        "image_light": "_static/kivi-mark.svg",
        "image_dark": "_static/kivi-mark.svg",
        "alt_text": "kivi",
    },
    "github_url": "https://github.com/gchensy/kivi",
    "navbar_start": ["navbar-logo", "version-badge"],
    "navbar_align": "left",
    "navbar_end": ["language-switch", "theme-switcher", "navbar-icon-links"],
    "header_links_before_dropdown": 5,
    "secondary_sidebar_items": ["page-toc"],
    "show_toc_level": 2,
    "footer_start": ["copyright"],
    "footer_center": [],
    "footer_end": ["sphinx-version"],
}
