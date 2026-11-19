Development
===========

Set up
------

Clone the repository and install it in editable mode with the test and documentation dependencies.

.. code-block:: bash

   git clone https://github.com/gchensy/kivi.git
   cd kivi
   python -m pip install -e '.[test,docs]'

Build and verify
----------------

.. code-block:: bash

   python -m pytest
   python -m sphinx -W -b doctest docs docs/_build/doctest
   python docs/build.py

The doctest builder runs every example in these pages and in the docstrings. The HTML build writes English and simplified Chinese editions under docs/_build/html. API signatures come from the installed package.

Translations
------------

The Chinese edition is generated from ``docs/locale/zh_CN/LC_MESSAGES/kivi.po``. Every paragraph of the English pages and of the public docstrings is one entry in that file. ``python docs/build.py`` stops when an entry is missing, fuzzy or no longer used.

After changing English text:

1. Run ``python docs/build.py --update-po``. New strings are added with an empty translation, and changed strings keep their old translation marked as fuzzy.
2. Translate the empty entries, review the fuzzy ones and delete their ``#, fuzzy`` line.
3. Run ``python docs/build.py`` again.

Both editions must keep the same page names, because the language switch links each page to its counterpart.

Publishing the documentation
----------------------------

The ``Documentation`` workflow on GitHub Actions runs the tests and builds both editions for every pull request. On the master branch it also publishes docs/_build/html to the gh-pages branch, which serves https://gchensy.github.io/kivi/.

Releasing to PyPI
-----------------

The ``Release`` workflow publishes kivi to PyPI when a GitHub release is published. It uses PyPI trusted publishing, so no API token is stored in the repository.

1. Set ``version`` in ``pyproject.toml`` to the new version, for example ``0.3.0``, and update :doc:`/release/index`.
2. Merge the change into the master branch.
3. Publish a GitHub release whose tag is ``v`` followed by the version, for example ``v0.3.0``.

The workflow stops if the tag does not match the version. Otherwise it runs the tests, builds the source distribution and the wheel, checks them with twine and uploads them.

Trusted publishing needs a one-time setup by an owner of the PyPI project: in the project's publishing settings, add a GitHub publisher for the repository ``gchensy/kivi`` with the workflow ``release.yml`` and the environment ``pypi``.

.. toctree::
   :hidden:

   legacy_review
