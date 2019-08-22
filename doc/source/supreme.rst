Supreme Court Corpus
====================

A collection of conversations from the U.S. Supreme Court Oral Arguments (51,498 utterances, from 204 cases). 

Distributed together with: `Echoes of power: Language effects and power differences in social interaction <https://www.cs.cornell.edu/~cristian/Echoes_of_power.html>`_. Cristian Danescu-Niculescu-Mizil, Bo Pang, Lillian Lee and Jon Kleinberg. WWW 2012


Dataset details
---------------

User-level information
^^^^^^^^^^^^^^^^^^^^^^

For each user, additional information include:

* is-justice: whether the user is a Justice 
* gender: gender of the user 


Utterance-level information
^^^^^^^^^^^^^^^^^^^^^^^^^^^

For each utterance, we provide:

* id: index of the utterance
* user: the user who author the utterance
* root: index of the conversation root of the utterance
* reply_to: index of the utterance to which this utterance replies to (None if the utterance is not a reply)
* timestamp: time of the utterance
* text: textual content of the utterance

Metadata for utterances may include:

* case: case number 
* justice-is-favorable: true if the Justice eventually vote for this side
* justice-vote: eventual vote from the Justice 
* side: side of the case

Note that some utterances may have only a subset of such information.  


Usage
-----

To download directly with ConvoKit: 

>>> from convokit import Corpus, download
>>> corpus = Corpus(filename=download("supreme-corpus"))


For some quick stats:

>>> len(corpus.get_utterance_ids()) 
51498
>>> len(corpus.get_usernames())
324
>>> len(corpus.get_conversations())
938


Additional note
---------------


Related links
^^^^^^^^^^^^^

1. `U.S. Supreme Court Oral Arguments <http://www.supremecourt.gov/oral_arguments/>`_.

2. Case outcome and vote data were extracted from the `Spaeth Supreme Court database <http://scdb.wustl.edu/>`_.

3. This corpus builds upon and enriches the data initially used in: `Computational analysis of the conversational dynamics of the United States Supreme Court <https://drum.lib.umd.edu/handle/1903/9999>`_. Timothy Hawes. Master's Thesis, 2009

Contact
^^^^^^^
Please email any questions to: cristian@cs.cornell.edu (Cristian Danescu-Niculescu-Mizil)

