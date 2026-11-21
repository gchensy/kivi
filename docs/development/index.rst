Development
===========

This section is for contributors. This page sets up a checkout and describes the checks and the rules for pull requests. :doc:`documentation` describes how the documentation is organized, written and translated, and :doc:`releasing` how a version is published.

Set up
------

Clone the repository and install it in editable mode with the test, documentation and lint dependencies, then install the pre-commit hook.

.. code-block:: bash

   git clone https://github.com/gchensy/kivi.git
   cd kivi
   python -m pip install -e '.[test,docs,lint]'
   pre-commit install

Build and verify
----------------

.. code-block:: bash

   ruff check .
   ruff format --check .
   python -m pytest
   python -m sphinx -W -b doctest docs docs/_build/doctest
   python docs/build.py

ruff checks the code against the rules in ``pyproject.toml`` and formats it. The pre-commit hook runs it before every commit, and the ``build`` check runs it on every pull request. A second job runs the tests with the oldest NumPy and pandas versions that kivi supports.

The doctest builder runs every example in these pages and in the docstrings. The HTML build writes English and simplified Chinese editions under docs/_build/html. API signatures come from the installed package.

Pull requests
-------------

Branch from the latest master and keep one topic per pull request. The master branch accepts changes only through pull requests, and only after the ``build`` check of the ``Documentation`` workflow has passed on a branch that is up to date with master. Every pull request is therefore tested together with everything merged before it. To update a branch, merge master into it. Merged branches are deleted automatically.

Pull requests that edit the same files are merged one after the other rather than developed in parallel. Two kinds of file change in almost every pull request and are set up so that they rarely conflict:

* Release notes. Describe each user-facing change as a bullet item in a new file ``docs/release/unreleased/<topic>.<category>.rst``, where the category is ``new``, ``changed``, ``fixed`` or ``removed``. Do not edit :doc:`/release/index` itself: the files appear there under the Unreleased heading.
* The translation catalog. When a merge or rebase stops on a conflict in ``kivi.po``, run ``python docs/build.py --merge-po``. It keeps the translations of both sides and adds the strings that are new after the merge. Translate those, then add the catalog to the commit.

A pull request that adds a module therefore only adds files: the module, its tests, a user guide page, an API page in ``docs/api/`` and a release note. The API reference and the user guide pick up new pages automatically. The pages that describe the whole library are updated together after a group of such pull requests has merged, as :doc:`documentation` describes.

.. toctree::
   :hidden:

   documentation
   releasing
   legacy_review
