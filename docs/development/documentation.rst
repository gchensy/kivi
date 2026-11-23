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
     - The pages of the getting started section and of the tutorial section, and :doc:`/user_guide/risk_workflow`
     - The installation, and worked examples on the bundled data from start to finish. They link to the topic guides for the details instead of repeating them.
   * - Topic guide
     - The other pages of the user guide
     - One step of a risk model: what it measures, an example and how to read the result. :doc:`/user_guide/datasets` describes the example data in the same way.
   * - Reference
     - The pages of the API reference
     - :doc:`/api/conventions`, the rules that every function follows; :doc:`/api/compatibility`, what a release may change; :doc:`/api/glossary`, the terms of the guides; and one page per public module, generated from its docstrings.
   * - Release notes
     - :doc:`/release/index` and :doc:`/release/migration`
     - What changed in each version, and how to move code from 0.9 to 1.0 and from the 0.2.x classes to their replacements. The notes of a released version are not edited. The migration guide is revised whenever a release changes what it describes.
   * - Contributor guide
     - The pages of this section
     - How to work on kivi. :doc:`legacy_review` records the review of the 0.2.x code that preceded the rewrite.

Where pages live
----------------

* A public module ``kivi.<module>`` has an API page ``docs/api/<module>.rst`` and usually a topic guide ``docs/user_guide/<topic>.rst``. The toctrees of the API reference and of the user guide list their pages in the order of a project and end with a glob, so a new page appears at the end without editing them. The integration moves it to its place, as Entry points below describes.
* The tutorial section, ``docs/tutorial``, follows one project over several chapters, listed in the order of the project by the toctree of its index page. Each chapter starts with a collapsed ``dropdown`` that repeats the code of the earlier chapters that it needs, so that the page runs on its own. Results that only simulated data can give, such as the outcomes of rejected applicants, go in an ``admonition`` titled ``Simulation check`` with the class ``kivi-check``, and the steps of the project never depend on them.
* The conventions, the compatibility promises and the glossary are ``docs/api/conventions.rst``, ``docs/api/compatibility.rst`` and ``docs/api/glossary.rst``. They come first in the API reference, and the index of the user guide links to the conventions and the glossary.
* Page names are part of the URLs, and the language switch links each page to the page of the same name in the other edition. Change the title of a page rather than renaming or removing it.
* The API page of a module and its topic guide have the same title. Its cards and its rows in the README use that title or a short form of it, such as WOE and IV.

Topic guides
------------

A topic guide follows the order of an analysis, so that readers can stop as soon as they have what they need. Its sections come in the order of the table below, with the titles that the table gives in inline code:

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Section
     - Contents
   * - Opening, without a heading
     - The question that the step answers and where it sits in a risk model, with links to the guides of the steps before and after it in the order of the user guide. The public functions that the page covers, each linked to its API entry. The opening has no examples.
   * - ``Definition``
     - The measure and its formula in a math block, before any example. Title it ``Definitions`` when it defines several measures, and leave it out when each tool of the page is defined in its own section.
   * - ``A worked example``
     - A few values that readers can check by hand, right after the definition.
   * - Body sections
     - One section for each task, tool or question of the step, in the order of a project, with an example on the bundled data and how to read its result. Title a body section with a noun phrase, such as ``Gains table``, or a gerund phrase, such as ``Choosing a cut-off``; imperative titles are for the steps of a tutorial. A body section may have subsections.
   * - ``Reading the result``
     - A table of the columns of the page's main result, the direction of the measure, its usual thresholds and what to check next. It follows the body sections whose examples print that result and comes before the body sections about other tools. On a page whose tools each return their own table, each body section describes its table instead.
   * - ``Special values``
     - How codes such as ``-1`` for "not applicable" are passed in ``special``, and which group they end up in.
   * - ``Missing values``
     - How missing values are counted, grouped or rejected.
   * - ``Sample weights``
     - What ``weights`` changes and what it leaves unchanged, and where weighted results differ from repeated rows.
   * - ``Limits``
     - What the method cannot tell, and what kivi leaves out.
   * - ``Errors and warnings``
     - The errors and warnings that readers are likely to meet, and how to resolve them.

Leave out the sections that do not apply, but keep the order and the titles of the others, so that the guides read alike. A variant that one or two sentences explain can stay in the body section that it belongs to, and an error that belongs to one task can be shown where that task is; a section of its own is for a variant or error that needs an example.

Section anchors are part of the URLs as well, and the READMEs link to some of them. Keep the title of a section that a README links to, and keep a label when its section moves.

Examples
--------

* Examples are doctests, and the ``build`` check runs them. Their output is real: when a change alters a printed table, run the code again and paste the new output instead of editing the numbers.
* Examples use the data described in :doc:`/user_guide/datasets`, or a few values that readers can check by hand. Random data comes from a generator with a fixed seed.
* Import each function from its module, as in ``from kivi.woe import woe_table``.
* Examples must not warn: a warning raised by a doctest, or by a chart or diagram recipe in docs/_plotting.py, docs/_diagrams.py or docs/_tutorial.py, fails the build. When an example is about a warning, catch it with ``warnings.catch_warnings(record=True)`` and print the message.
* All the examples of a page share one namespace, so a section can use what an earlier section defined. When sections move, check that every name is still defined before it is used.

Charts
------

Charts and diagrams are drawn when the documentation is built. A ``plot`` directive calls a ``figure_*`` function, which recomputes the example shown just above it, so change the two together:

* A chart, from ``docs/_plotting.py`` and called as ``kp.figure_*()``, plots a result on the bundled data, such as the WOE of each group or the vintage curves.
* A chart of the tutorial, from ``docs/_tutorial.py`` and called as ``kt.figure_*()``, rebuilds the state of the project once and draws it with the functions of ``docs/_plotting.py``.
* A diagram, from ``docs/_diagrams.py`` and called as ``kd.figure_*()``, shows how a function computes its result: the input rows, the call and the result table, with the numbers that kivi returns. A topic guide shows its main computation in a diagram right after the worked example, or after the first example of the body section that the computation belongs to.

Text inside a chart or a diagram is limited to kivi's column and parameter names, data values and formulas written with them, so that one image serves both editions. The caption explains the image and the alternative text describes it; both are translated. A recipe that computes numbers apart from the function it shows checks them against that function, so that a change of the function fails the build instead of leaving the image out of date.

Every diagram gives a color the same meaning:

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - Color
     - Meaning
   * - One tint per category, a blue ramp from light to dark for intervals, gray for missing values
     - Groups and bins, in the input and in the result
   * - Red
     - Events, the rows with target 1
   * - Amber
     - The result that the diagram is about, and the edges and thresholds
   * - Blue and teal
     - The reference sample and the current sample

A pull request that adds a module describes the charts and diagrams for its page in its description. They are added when the page is integrated, together with the entry points.

Names and links
---------------

* Write argument and column names as inline code, such as ``special`` or ``event_rate``.
* In the topic guides and the tutorials, link the first mention of a kivi function, class or module in each section to its API entry. Use the ``:func:``, ``:class:``, ``:exc:``, ``:meth:`` and ``:mod:`` roles with the full name and a leading tilde, such as ``:func:`~kivi.woe.woe_table```, so that only the name is shown. Keep the module in the text where it tells two functions apart, as in ``:func:`kivi.binning.cut``` next to ``pandas.cut``. Later mentions in the same section stay inline code.
* ``woe``, ``iv``, ``psi``, ``ks`` and ``gini`` are the names of functions and of columns. Link them only where they mean the function.
* Names inside examples are code and are not linked.
* Link to another page with the ``:doc:`` role and to a section of another page with the ``:ref:`` role. Give a section a label only when a link points to it.

API reference
-------------

* :doc:`/api` lists the conventions, the compatibility page, the glossary and then the module pages in the order of the user guide, and ends with a glob for the pages that are not integrated yet. ``api/datasets`` comes last through the glob, so that the glob always matches a page.
* A module page starts with its title, a ``module`` directive, which gives ``:mod:`` links their target, and an introduction: one or two sentences on what the module is for, with links to its main functions, and a sentence that links to its topic guide. The entries follow. For example:

  .. code-block:: rst

     Binning
     =======

     .. module:: kivi.binning

     Fit bin edges on a reference sample and group every sample with the same edges.
     :func:`~kivi.binning.bin_edges` fits them from the values alone, ...

     See :doc:`/user_guide/binning` for examples and how to choose the edges.

     .. autosummary::

        bin_edges
        cut

     .. autofunction:: bin_edges

     .. autofunction:: cut

* The API page of a module documents every name in the module's ``__all__``: first in an ``autosummary`` table, then each with its own ``autofunction``, ``autoclass``, ``autoexception`` or ``automethod`` entry, in the same order. ``python docs/build.py`` stops when a name has no such entry.
* Docstrings follow the numpydoc conventions and are validated with every check enabled, so a missing section, such as See Also or Examples, fails the build. Their paragraphs are translated like those of the pages.
* :doc:`/api/conventions` holds the rules that every module follows, such as the order of the arguments and the meaning of ``weights``. A guide or a docstring states a rule of its own module and links to the conventions for the shared ones. :doc:`/api/glossary` defines each term once and links to the guide that uses it.

Translations
------------

The Chinese edition is generated from ``docs/locale/zh_CN/LC_MESSAGES/kivi.po``. Every paragraph of the English pages and of the public docstrings is one entry in that file. ``python docs/build.py`` stops when an entry is missing, fuzzy or no longer used.

After changing English text:

1. Run ``python docs/build.py --update-po``. New strings are added with an empty translation, and changed strings keep their old translation marked as fuzzy.
2. Translate the empty entries, review the fuzzy ones and delete their ``#, fuzzy`` line.
3. Run ``python docs/build.py`` again.

In a translation, put a space or Chinese punctuation on both sides of inline markup such as inline code and links, or the markup is not recognized. Translate each term as the catalog already translates it, so that the edition keeps one vocabulary.

A change that only replaces markup, such as inline code by a link, leaves the translations valid. Apply it to the source strings and to their translations in ``kivi.po`` before step 1, so that the entries do not turn fuzzy.

Entry points
------------

Some pages describe the library as a whole: ``README.md`` and ``README.zh-CN.md``, the home page, the index of the user guide and :doc:`/user_guide/risk_workflow`. A pull request that adds a module leaves them alone. They are updated together after a group of such pull requests has merged, and that update rewrites them for the library as it now is instead of only appending to them:

* the title, the introduction and the steps on the home page;
* the cards on the home page and on the index of the user guide, and the order of the user guide;
* the order of the API reference;
* the links to the previous and the next step in the openings of the guides next to a new one;
* the terms of :doc:`/api/glossary` and the rules of :doc:`/api/conventions` that a new module adds;
* the tables and the roadmap of both READMEs;
* any sentence in :doc:`/release/migration` about functionality that is not ported yet.

A pull request that adds a module lists in its description the glossary terms and the conventions that it needs, as it does for its charts.

``python docs/build.py`` lists every public function that the API tables of the READMEs leave out, and every user guide page without a card on the index of the user guide or a link in the READMEs. ``python docs/build.py --release`` refuses to run until the list is empty.

Publishing the documentation
----------------------------

The ``Documentation`` workflow on GitHub Actions runs the tests and builds both editions for every pull request. On the master branch it also publishes docs/_build/html to the gh-pages branch, which serves https://gchensy.github.io/kivi/.
