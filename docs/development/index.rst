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

The ``Release`` workflow publishes kivi to PyPI when a version tag is pushed. It uses PyPI trusted publishing, so no API token is stored in the repository.

1. Set ``version`` in ``pyproject.toml`` to the new version, for example ``0.3.1``, and update :doc:`/release/index`.
2. Merge the change into the master branch.
3. Tag the merged commit with ``v`` followed by the version and push the tag:

   .. code-block:: bash

      git switch master
      git pull
      git tag v0.3.1
      git push origin v0.3.1

4. Approve the deployment to the ``pypi`` environment when GitHub asks for it in the workflow run.

The workflow stops before anything is uploaded if the tag does not match the version or points at a commit that is not on master. Otherwise it runs the tests, builds the source distribution and the wheel, checks them with twine and uploads them to PyPI. Only then does it create the GitHub release, with the distributions attached and notes generated from the merged pull requests. A version such as ``0.4.0rc1`` is marked as a pre-release.

The repository protects this path in three places:

* Only repository administrators can create, move or delete ``v*`` tags.
* The ``pypi`` environment accepts deployments from ``v*`` tags only and waits for a maintainer's approval.
* PyPI trusts only the ``release.yml`` workflow of ``gchensy/kivi`` running in the ``pypi`` environment.

PyPI never accepts the same version twice. If a run fails before the upload, delete the tag with ``git push origin :refs/tags/v0.3.1``, fix the problem and tag again. If the upload succeeded, release the fix under a new version.

.. toctree::
   :hidden:

   legacy_review
