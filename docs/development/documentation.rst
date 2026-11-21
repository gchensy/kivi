Writing documentation
=====================

The English and the simplified Chinese editions are built from the same pages and docstrings. This page describes how they are organized and written. :doc:`index` covers the checks that every pull request runs.

Kinds of pages
--------------

Apart from the home page and the index page of each section, every page is one of five kinds, and its kind decides what it contains.

.. list-table::
   :header-rows: 1
   :widths: 20 30 50

   * - Kind
     - Pages
     - Contents
   * - Tutorial
     - The pages of the getting started section, and :doc:`/user_guide/risk_workflow`
     - The installation, and worked examples on the bundled data from start to finish. They link to the topic guides for the details instead of repeating them.
   * - Topic guide
     - The other pages of the user guide
     - One step of a risk model: what it measures, an example and how to read the result. :doc:`/user_guide/datasets` describes the example data in the same way.
   * - Reference
     - The pages of the API reference
     - One page per public module, generated from its docstrings.
   * - Release notes
     - :doc:`/release/index` and :doc:`/release/migration`
     - What changed in each version, and how to move code from the 0.2.x classes to their replacements. The notes of a released version are not edited. The migration guide is revised whenever a release changes what it describes.
   * - Contributor guide
     - The pages of this section
     - How to work on kivi. :doc:`legacy_review` records the review of the 0.2.x code that preceded the rewrite.

Where pages live
----------------

* A public module ``kivi.<module>`` has an API page ``docs/api/<module>.rst`` and usually a topic guide ``docs/user_guide/<topic>.rst``. The toctrees of the API reference and of the user guide end with a glob, so a new page appears without editing them.
* Page names are part of the URLs, and the language switch links each page to the page of the same name in the other edition. Change the title of a page rather than renaming or removing it.
* The API page of a module and its topic guide have the same title. Its cards and its rows in the README use that title or a short form of it, such as WOE and IV.

Topic guides
------------

A topic guide follows the order of an analysis, so that readers can stop as soon as they have what they need:

1. An opening paragraph: the question that the step answers, where it sits in a risk model, and links to the steps before and after it.
2. The definition, with the formula in a math block when there is one.
3. A worked example on the bundled data.
4. How to read the result: the direction of the measure, the usual thresholds and what to check next.
5. Variants, each in a section with a fixed title: ``Special values``, ``Missing values`` or ``Sample weights``.
6. The errors and warnings that readers are likely to meet, and how to resolve them.

Leave out the parts that do not apply, but keep their order and titles, so that the guides read alike.

Examples
--------

* Examples are doctests, and the ``build`` check runs them. Their output is real: when a change alters a printed table, run the code again and paste the new output instead of editing the numbers.
* Examples use the data described in :doc:`/user_guide/datasets`, or a few values that readers can check by hand. Random data comes from a generator with a fixed seed.
* Import each function from its module, as in ``from kivi.woe import woe_table``.
* Examples must not warn: a warning raised by a doctest or by a chart recipe in docs/_plotting.py fails the build. When an example is about a warning, catch it with ``warnings.catch_warnings(record=True)`` and print the message.

Charts
------

Charts are drawn when the documentation is built. A ``plot`` directive calls a ``figure_*`` function in ``docs/_plotting.py``, which recomputes the example shown just above the chart, so change the two together. Text inside a chart is limited to kivi's column and parameter names and to data values, so that one image serves both editions. The caption and the alternative text are translated.

A pull request that adds a module describes the charts for its page in its description. They are added to ``docs/_plotting.py`` when the page is integrated, together with the entry points.

Names and links
---------------

* Write function, argument and column names as inline code, such as ``woe_table``, ``special`` or ``event_rate``.
* In a topic guide, link the first mention of each kivi function to its API entry. Use the ``:func:`` role with the full name, such as ``kivi.woe.woe_table``, and a leading tilde, so that only the function name is shown.
* Link to another page with the ``:doc:`` role and to a section of another page with the ``:ref:`` role. Give a section a label only when a link points to it.

API reference
-------------

* The API page of a module documents every name in the module's ``__all__``: first in an ``autosummary`` table, then each with its own ``autofunction``, ``autoclass``, ``autoexception`` or ``automethod`` entry, in the same order. ``python docs/build.py`` stops when a name has no such entry.
* Docstrings follow the numpydoc conventions and are validated with every check enabled, so a missing section, such as See Also or Examples, fails the build. Their paragraphs are translated like those of the pages.

Translations
------------

The Chinese edition is generated from ``docs/locale/zh_CN/LC_MESSAGES/kivi.po``. Every paragraph of the English pages and of the public docstrings is one entry in that file. ``python docs/build.py`` stops when an entry is missing, fuzzy or no longer used.

After changing English text:

1. Run ``python docs/build.py --update-po``. New strings are added with an empty translation, and changed strings keep their old translation marked as fuzzy.
2. Translate the empty entries, review the fuzzy ones and delete their ``#, fuzzy`` line.
3. Run ``python docs/build.py`` again.

In a translation, put a space or Chinese punctuation on both sides of inline markup such as inline code and links, or the markup is not recognized. Translate each term as the catalog already translates it, so that the edition keeps one vocabulary.

Entry points
------------

Some pages describe the library as a whole: ``README.md`` and ``README.zh-CN.md``, the home page, the index of the user guide and :doc:`/user_guide/risk_workflow`. A pull request that adds a module leaves them alone. They are updated together after a group of such pull requests has merged, and that update rewrites them for the library as it now is instead of only appending to them:

* the title, the introduction and the steps on the home page;
* the cards on the home page and on the index of the user guide, and the order of the user guide;
* the tables and the roadmap of both READMEs;
* any sentence in :doc:`/release/migration` about functionality that is not ported yet.

``python docs/build.py`` lists every public function that the API tables of the READMEs leave out, and every user guide page without a card on the index of the user guide or a link in the READMEs. ``python docs/build.py --release`` refuses to run until the list is empty.

Publishing the documentation
----------------------------

The ``Documentation`` workflow on GitHub Actions runs the tests and builds both editions for every pull request. On the master branch it also publishes docs/_build/html to the gh-pages branch, which serves https://gchensy.github.io/kivi/.
