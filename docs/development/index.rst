Development
===========

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

Examples must not warn: a warning raised by a doctest or by a chart recipe in docs/_plotting.py fails the build. When an example is about a warning, catch it with ``warnings.catch_warnings(record=True)`` and print the message.

Pull requests
-------------

Branch from the latest master and keep one topic per pull request. The master branch accepts changes only through pull requests, and only after the ``build`` check of the ``Documentation`` workflow has passed on a branch that is up to date with master. Every pull request is therefore tested together with everything merged before it. To update a branch, merge master into it. Merged branches are deleted automatically.

Pull requests that edit the same files are merged one after the other rather than developed in parallel. Two kinds of file change in almost every pull request and are set up so that they rarely conflict:

* Release notes. Describe each user-facing change as a bullet item in a new file ``docs/release/unreleased/<topic>.<category>.rst``, where the category is ``new``, ``changed``, ``fixed`` or ``removed``. Do not edit :doc:`/release/index` itself: the files appear there under the Unreleased heading.
* The translation catalog. When a merge or rebase stops on a conflict in ``kivi.po``, run ``python docs/build.py --merge-po``. It keeps the translations of both sides and adds the strings that are new after the merge. Translate those, then add the catalog to the commit.

A pull request that adds a module therefore only adds files: the module, its tests, a user guide page, an API page in ``docs/api/`` and a release note. The API reference and the user guide pick up new pages automatically. The README tables, the cards on the home page and on the user guide index, and the risk modeling workflow are updated together after a group of such pull requests has merged.

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

The ``Release`` workflow publishes kivi to PyPI when a version tag is pushed. It uses PyPI trusted publishing, so no API token is stored in the repository.

1. Set ``version`` in ``pyproject.toml`` to the new version, for example ``0.3.1``. Run ``python docs/build.py --release 0.3.1``, which moves the release-note files into a section of :doc:`/release/index` dated today, and translate the new heading.
2. Merge the change into the master branch.
3. Tag the merged commit with ``v`` followed by the version and push the tag:

   .. code-block:: bash

      git switch master
      git pull
      git tag v0.3.1
      git push origin v0.3.1

Pushing the tag starts the upload without an approval step, so push it only when the release is ready. The workflow stops before anything is uploaded if the tag does not match the version or points at a commit that is not on master. Otherwise it runs the tests, builds the source distribution and the wheel, checks them with twine and uploads them to PyPI. Only then does it create the GitHub release, with the distributions attached and notes generated from the merged pull requests. A version such as ``0.4.0rc1`` is marked as a pre-release.

The repository protects this path in three places:

* Only repository administrators can create, move or delete ``v*`` tags.
* The ``pypi`` environment accepts deployments from ``v*`` tags only.
* PyPI trusts only the ``release.yml`` workflow of ``gchensy/kivi`` running in the ``pypi`` environment.

PyPI never accepts the same version twice. If a run fails before the upload, delete the tag with ``git push origin :refs/tags/v0.3.1``, fix the problem and tag again. If the upload succeeded, release the fix under a new version.

.. toctree::
   :hidden:

   legacy_review
