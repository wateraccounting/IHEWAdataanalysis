===========
IHEWAreport
===========

This is the documentation of **IHEWAreport**.

**IHEWAreport** is a Reportion of tools to write standard report.
This project is fully developed by Water Accounting team at IHE-Delft.

`MiKTeX <https://miktex.org/>`_ is required to generate tex and pdf file.

    - .py  -> latexmk  -> .tex & .run.xml
    - .tex -> pdflatex -> .bcf
    - .bcf -> biber    -> .bbl
    - .bbl -> pdflatex -> .pdf 1st time
    - .bbl -> pdflatex -> .pdf 2ed time, make sure

.. warning::

    Set gdal executable files to system "PATH".

    Must contain **config.yml** file,
    see `config.yml-example <https://github.com/wateraccounting/IHEWAreport/tree/master/examples>`_.


Installation
============

Install from pip
^^^^^^^^^^^^^^^^

IHEWAreport Python package is hosted on `Python Package Index (PyPI)
<https://pypi.org/project/IHEWAreport/>`_.

Install from pip.

.. code-block:: console

    $ pip install IHEWAreport

Install from source code
^^^^^^^^^^^^^^^^^^^^^^^^

Download source code from `Github
<https://github.com/wateraccounting/IHEWAreport>`_.

.. code-block:: console

    $ git clone https://github.com/wateraccounting/IHEWAreport.git
    $ cd IHEWAreport

Install from source code.

.. code-block:: console

    $ python setup.py install

Uninstall IHEWAreport
^^^^^^^^^^^^^^^^^^^^^

.. code-block:: console

    $ pip uninstall IHEWAreport


Docker
======

Set "System -> Base Memory" to max capacity in Virtual Machine.

Set "Shared Folders" in Virtual Machine.

- Folder Path: D:\\IHEWAreport
- Folder Name: d/IHEWAreport
- Auto-mount: Yes
- Make Permanent: Yes

Download source code from `Github
<https://github.com/wateraccounting/IHEWAreport>`_.

.. code-block:: console

    $ git clone https://github.com/wateraccounting/IHEWAreport.git
    $ cd IHEWAreport

Restart Docker Virtual Machine image.

.. code-block:: console

    $ docker-machine restart

Pull the IHEWAreport image.

.. code-block:: console

    $ docker pull wateraccounting/ihewareport

Or build from source code.

.. code-block:: console

    $ docker build -t wateraccounting/ihewareport .

Check images.

.. code-block:: console

    $ docker images

Check ip address.

.. code-block:: console

    $ docker-machine env
    export DOCKER_HOST="tcp://192.168.99.100:2376"

Run the image with Jupyter Notebook. To access Jupyter http://192.168.99.100:8888/

.. code-block:: console

    $ docker run -it --name ihewareport -p 8888:8888 -v /d/IHEWAreport:/notebooks wateraccounting/ihewareport

Check running image (container), in the new cmd window.

.. code-block:: console

    $ docker container list

Access to running image, in the new cmd window.

.. code-block:: console

    $ docker exec -it ihewareport bash

    $ cd /notebooks/

    $ python3 setup.py install
    $ python3 ./examples/ex_CFSR_GRIB.py

Clean running cache.

.. code-block:: console

    $ docker system prune -f && docker volume prune -f && docker container prune -f

Remove image.

.. code-block:: console

    $ docker rmi wateraccounting/ihewareport


Use
===

Examples can be found at `examples
<https://github.com/wateraccounting/IHEWAreport/tree/master/examples>`_.

.. code-block:: python

    import os
    import IHEWAreport

    path = os.getcwd()
    config = 'config.yml'
    file_config = os.path.join(path, config)

    if os.path.exists(file_config):
        IHEWAreport.Report(workspace=path,
                           config=config)
)


Development
===========

In the PyCharm, change "Project Structure -> Source Folders" to "src"

Download source code from `Github
<https://github.com/wateraccounting/IHEWAreport>`_.

.. code-block:: console

    $ git clone https://github.com/wateraccounting/IHEWAreport.git
    $ cd IHEWAreport

From the root of the project

.. code-block:: console

    $ python setup.py --version

Format scripts by PEP8

.. code-block:: console

    $ autopep8 --in-place --aggressive src/IHEWAreport/base/base.py

Flake8, pre-commit

.. code-block:: console

    $ pre-commit install

    $ pre-commit run --all-files
    [INFO] Initializing environment for git://github.com/pre-commit/pre-commit-hooks.
    [INFO] Initializing environment for https://github.com/pre-commit/mirrors-isort.
    [INFO] Installing environment for git://github.com/pre-commit/pre-commit-hooks.
    [INFO] Once installed this environment will be reused.
    [INFO] This may take a few minutes...
    [INFO] Installing environment for https://github.com/pre-commit/mirrors-isort.
    [INFO] Once installed this environment will be reused.
    [INFO] This may take a few minutes...

Unit test

.. code-block:: console

    $ python setup.py test

Read the Docs

.. code-block:: console

    $ python setup.py doctest

    $ python setup.py docs

Upload to PyPI

1. In IDE, **commit** the changes "**v0.0.1**"
2. In IDE, **Version Control -> Log**, select this commit
3. In IDE, add version tag, select **VCS -> Git -> tag**
4. In IDE, **Tag window -> Tag Name**, type "**v0.0.1**"

5. In cmd, build package, type ``python setup.py sdist bdist_wheel``
6. In cmd, validate build, type ``twine check dist/IHEWAreport-0.0.1*``
7. In cmd, upload build, type ``twine upload dist/IHEWAreport-0.0.1*``

8. In IDE, **push** the commit, with Tag label: "*HEAD*", "*master*", "*v0.0.1*"
9. In Github, select **Release** to "create a new release" or "Draft a new release"
10. In Github, **Tag version**, type "**v0.0.1**"
11. In Github, **@ Target**, select this commit
12. In Github, **Publish release**


Templates
=========

Details can be found at `Templates
<https://IHEWAreport.readthedocs.io/en/latest/templates.html>`_.


Code of Conduct
===============

  - Be friendly and patient
  - Be welcoming
  - Be considerate
  - Be respectful
  - Be careful in the words that you choose
  - When we disagree, try to understand why


Contents
========

.. toctree::
   :maxdepth: 4

   License <license>
   Authors <authors>
   Contributing <contributing>
   Changelog <changelog>
   Templates <templates>
   Module Reference <api/modules>


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

.. _toctree: http://www.sphinx-doc.org/en/master/usage/restructuredtext/directives.html
.. _reStructuredText: http://www.sphinx-doc.org/en/master/usage/restructuredtext/basics.html
.. _references: http://www.sphinx-doc.org/en/stable/markup/inline.html
.. _Python domain syntax: http://sphinx-doc.org/domains.html#the-python-domain
.. _Sphinx: http://www.sphinx-doc.org/
.. _Python: http://docs.python.org/
.. _Numpy: http://docs.scipy.org/doc/numpy
.. _SciPy: http://docs.scipy.org/doc/scipy/reference/
.. _matplotlib: https://matplotlib.org/contents.html#
.. _Pandas: http://pandas.pydata.org/pandas-docs/stable
.. _Scikit-Learn: http://scikit-learn.org/stable
.. _autodoc: http://www.sphinx-doc.org/en/stable/ext/autodoc.html
.. _Google style: https://github.com/google/styleguide/blob/gh-pages/pyguide.md#38-comments-and-docstrings
.. _NumPy style: https://numpydoc.readthedocs.io/en/latest/format.html
.. _classical style: http://www.sphinx-doc.org/en/stable/domains.html#info-field-lists
