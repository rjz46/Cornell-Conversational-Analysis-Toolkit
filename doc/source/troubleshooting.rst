Troubleshooting FAQ
===================

General checks
^^^^^^^^^^^^^^
- Check that you are using the latest version of ConvoKit
- Verify that your installed package dependencies for ConvoKit satisfy `ConvoKit's versioning requirements <https://github.com/CornellNLP/Cornell-Conversational-Analysis-Toolkit/blob/master/requirements.txt>`_
- If possible, use a Unix system, i.e. Mac OS or the Linux distros. We advise against using Windows, but Windows users may consider using the Windows subsystem for Linux (WSL) instead.

Issues
^^^^^^

**OSError: [E050] Can't find model 'en'. It doesn't seem to be a shortcut link, a Python package or a valid path to a directory.**

As mentioned in the installation instructions, one needs to run "python -m spacy download en" so that a model 'en' exists.

However, there is a secondary issue specific to Windows machines:

-----------------------------

**python -m spacy download en** appears successful but actually fails to link the downloaded model to spaCy [Windows]

.. image:: img/windows-failed-spacy-link.jpeg
.. image:: img/windows-failed-spacy-load.jpeg

The output from the command suggests that linking is successful and that *spacy.load('en')* should succeed. However, a closer inspection of the first set of outputs reveals an error message: "You do not have sufficient privilege to perform this operation."

The operation referred to is the linking of 'en'. This issue has been raised `here <https://github.com/explosion/spaCy/issues/1283>`_ and has been acknowledged as a bug.

The solution is to either do the installation in a virtualenv (where the privileges required for linking is lower) or run powershell as administrator.

-----------------------------

**error: Microsoft Visual C++ 14.0 is required.** when installing SpaCy [Windows]

SpaCy, one of ConvoKit's dependencies, itself has an assortment of dependencies (e.g. murmurhash, cytoolz.) On Windows, these must be built using Microsoft Visual C++ build tools to be built properly.

The build tools can be downloaded `from Microsoft here <https://visualstudio.microsoft.com/downloads/#build-tools-for-visual-studio-2019>`_. Take care to not `confuse these tools with other Microsoft distributables <https://github.com/explosion/spaCy/issues/2441>`_.

-----------------------------

**error: command 'gcc' failed with exit status 1** [Mac OS]

This is an error encountered when installing the SpaCy dependency for ConvoKit on MacOS. The solution is to link the required C++ standard library explicitly, like so:

>>> CFLAGS=-stdlib=libc++ python3 -m pip install convokit


