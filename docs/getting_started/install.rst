Installation
============

kivi requires Python 3.10 or newer, NumPy 1.24 or newer and pandas 2.0 or newer. pip installs NumPy and pandas automatically.

.. tab-set::

   .. tab-item:: PyPI

      Install the latest release:

      .. code-block:: bash

         python -m pip install kivi

   .. tab-item:: GitHub

      Install the development version directly from the repository:

      .. code-block:: bash

         python -m pip install "git+https://github.com/gchensy/kivi"

   .. tab-item:: Local checkout

      Clone the repository and install it from the checkout:

      .. code-block:: bash

         git clone https://github.com/gchensy/kivi.git
         cd kivi
         python -m pip install .

Check the installed version:

.. code-block:: bash

   python -c "import kivi; print(kivi.__version__)"

To work on kivi itself, use an editable install with the test and documentation dependencies. See :doc:`/development/index`.

.. note::

   The 0.2.x releases have a different API. Read :doc:`/release/migration` before upgrading existing code.
