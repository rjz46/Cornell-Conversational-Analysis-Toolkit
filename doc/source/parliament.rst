Parliament Question Time Corpus
===============================

A collections of questions and answers from parliamentary question periods in the British House of Commons from May 1979 to December 2016 (433,787 utterances). 

Distributed together with:
`Asking Too Much? The Rhetorical Role of Questions in Political Discourse <https://www.cs.cornell.edu/~cristian/Asking_too_much.html>`_. Justine Zhang, Arthur Spirling, Cristian Danescu-Niculescu-Mizil. EMNLP 2017.


Dataset details
---------------

User-level information
^^^^^^^^^^^^^^^^^^^^^^

The users in the dataset are members of the Parliament. For each user, the dataset further includes the following metadata:

* name: name of the user
* member_start: start time of the user as the member of the Parliament 
* member_end: end time of the user as the member of the Parliament
* changed_parties: whether the user has changed parties
* first_party: first party the user is affiliated with
* last_party: last party the user is affiliated with


Utterance-level information
^^^^^^^^^^^^^^^^^^^^^^^^^^^

Each question or answer is viewed as an utterance. For each utterance, we provide:

* id: index of the utterance
* user: the user who author the utterance
* root: index of the conversation root of the utterance
* reply_to: index of the utterance to which this utterance replies to (None if the utterance is not a reply)
* timestamp: time of the utterance
* text: textual content of the utterance

Additional metadata include:

* is_question: whether the utterance is a question
* is_answer: whether the utterance is an answer to a question
* pair_idx: index of the question-answer pair
* is_incubent: whether the user is an Incumbent
* is_minister: whether the user is a Minister
* is_oppn: whether the user is from the opposition party
* party: party affiliation of the user
* parsed: parsed version of the utterance text, represented as a SpaCy Doc


Usage
-----

To download directly with ConvoKit: 

>>> from convokit import Corpus, download
>>> corpus = Corpus(filename=download("parliament-corpus"))


For some quick stats:

>>> len(corpus.get_utterance_ids()) 
433787
>>> len(corpus.get_usernames())
1978
>>> len(corpus.get_conversation_ids())
216894


Additional note
---------------

See this `example notebook <https://github.com/CornellNLP/Cornell-Conversational-Analysis-Toolkit/blob/master/examples/question-typology/parliament_questions_example.ipynb>`_. for an example of how to group questions in this dataset according to their rhetorical roles.  
