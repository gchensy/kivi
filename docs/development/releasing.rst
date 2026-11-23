Releasing to PyPI
=================

The ``Release`` workflow publishes kivi to PyPI when a version tag is pushed. It uses PyPI trusted publishing, so no API token is stored in the repository.

1. Set ``version`` in ``pyproject.toml`` to the new version, for example ``0.3.1``. Run ``python docs/build.py --release 0.3.1``, which moves the release-note files into a section of :doc:`/release/index` dated today, and translate the new heading.

   The command changes nothing while ``python docs/build.py`` still reports entry points to update, as described in :doc:`documentation`.
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

Version numbers
---------------

Each release follows the promises of :doc:`/api/compatibility`. From 1.0:

* A release that only fixes errors is a patch release, such as 1.3.1.
* A release that adds to the API, deprecates a part of it or raises a minimum version is a minor release, such as 1.4.
* A release that removes or changes what earlier releases deprecated is a major release, such as 2.0.

A fix of a wrong result can go into a patch or a minor release, with a ``fixed`` fragment that says what was wrong.

Deprecations
------------

A deprecated name, argument or behavior keeps working and warns with ``FutureWarning`` through ``_deprecation.warn``, which points the warning at the caller's line however deep the check sits. The message names what is deprecated and says what to use instead. ``tests/test_deprecations.py`` checks each warning with ``pytest.warns(FutureWarning, match=...)`` and that it points at the test's own line. The code, the documentation and the README examples must not trigger a deprecation themselves, because the tests and the doctests turn warnings into errors.

The ``changed`` fragment of the release names the deprecation and its replacement. The major release that removes it lists the removal in a ``removed`` fragment and deletes its test.

Minimum versions
----------------

A minimum version of Python, NumPy or pandas rises only in a minor release, and a ``changed`` fragment of the release before announces it. The release that raises it changes ``requires-python`` or ``dependencies`` in ``pyproject.toml``, the ``minimum-versions`` job of ``.github/workflows/docs.yml``, which tests the oldest supported versions, and the versions on :doc:`/getting_started/install` and :doc:`/api/compatibility` together.
