from importlib.metadata import version as distribution_version

project = "kivi"
author = "Chensy"
release = distribution_version("kivi")
version = release
extensions = ["sphinx.ext.autodoc", "sphinx.ext.doctest", "numpydoc"]
numpydoc_show_class_members = False
language = "en"
locale_dirs = ["locale/"]
gettext_compact = "kivi"
exclude_patterns = ["_build", "_templates"]
templates_path = ["_templates"]
html_theme = "pydata_sphinx_theme"
html_title = f"kivi {release}"
html_theme_options = {
    "github_url": "https://github.com/gchensy/kivi",
    "navbar_end": ["language-switch", "theme-switcher", "navbar-icon-links"],
}
