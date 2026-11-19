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

Publishing
----------

The ``Documentation`` workflow on GitHub Actions runs the tests and builds both editions for every pull request. On the master branch it also publishes docs/_build/html to the gh-pages branch, which serves https://gchensy.github.io/kivi/.

.. toctree::
   :hidden:

   legacy_review
