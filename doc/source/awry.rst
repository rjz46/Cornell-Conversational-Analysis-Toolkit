Conversations Gone Awry Dataset
===============================

A collection of conversations from Wikipedia talk pages that derail into personal attacks (4,188 conversations, 30,021 comments). 

Distributed together with: 

`Conversations gone awry: Detecting early signs of conversational failure <https://www.cs.cornell.edu/~cristian/Conversations_gone_awry_files/conversations_gone_awry.pdf>`_. Justine Zhang, Jonathan P. Chang, Cristian Danescu-Niculescu-Mizil, Lucas Dixon, Yiqing Hua, Nithum Thain, Dario Taraborelli. ACL 2018. 

and

Trouble on the Horizon: Forecasting the Derailment of Online Conversations as they Develop. Jonathan P. Chang and Crisitan Danescu-Niculescu-Mizil. EMNLP 2019.

Dataset details
---------------

User-level information
^^^^^^^^^^^^^^^^^^^^^^

Users in this dataset are Wikipedia editors; their account names are taken as the user names. 

Utterance-level information
^^^^^^^^^^^^^^^^^^^^^^^^^^^

Each conversational turn on the talk page is viewed as an utterance. For each utterance, we provide:

* id: index of the utterance
* user: the user who author the utterance
* root: index of the conversation root of the utterance
* reply_to: index of the utterance to which this utterance replies to (None if the utterance is not a reply)
* timestamp: time of the utterance
* text: textual content of the utterance

Metadata for each utterance include:

* is_section_header: whether the utterance is a conversation "title" or "subject" as seen on the original talk page (if true, this utterance should be ignored when doing any NLP tasks)
* comment_has_personal_attack: whether this comment was judged by 3 crowdsourced annotators to contain a personal attack
* parsed: parsed version of the utterance text, represented as a SpaCy Doc


Conversational-level information
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Metadata for each conversation include:

* page_title: the title of the talk page the comment came from
* page_id: the unique numerical ID of the talk page the comment came from
* pair_id: the conversation ID (root) of the conversation that this comment's conversation is paired with
* conversation_has_personal_attack: whether any comment in this comment's conversation contains a personal attack according to crowdsourced annotators
* verified: whether the personal attack label has been double-checked by an internal annotator and confirmed to be correct 
* pair_verified: whether the personal attack label for the paired conversation has been double-checked by an internal annotator and confirmed to be correct 
* annotation_year: which round of annotation the conversation's label came from. Possible values are "2018" for the first annotation round and "2019" for the second annotation round.
* split: which split (train, val, or test) this conversation was used in for the experiments described in "Trouble on the Horizon" (not applicable to results from "Conversations Gone Awry", which reports leave-one-out accuracies).


Usage
-----

To download directly with ConvoKit: 

>>> from convokit import Corpus, download
>>> corpus = Corpus(filename=download("conversations-gone-awry-corpus"))


For some quick stats:

>>> len(corpus.get_utterance_ids()) 
30021
>>> len(corpus.get_usernames())
8069
>>> len(corpus.get_conversation_ids())
4188


Additional note
---------------

This data was collected from late 2017 to early 2018 and was annotated in two rounds: one round in April 2018 (for "Conversations Gone Awry") and another in February 2019 (for "Trouble on the Horizon").


Related links
^^^^^^^^^^^^^

* Fun: Guess whether a `conversation will go awry <https://awry.infosci.cornell.edu/>`_. 

* `Wikipedia editors' talk Pages <http://en.wikipedia.org/wiki/Wikipedia:Talk_page_guidelines>`_.


Contact
^^^^^^^

Please email any questions to: cristian@cs.cornell.edu (Cristian Danescu-Niculescu-Mizil)







