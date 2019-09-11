Conversations Gone Awry Dataset [Reddit CMV version]
====================================================

A collection of conversations from the ChangeMyView (CMV) subreddit that derail into personal attacks (6,842 conversations, 42,964 comments). 

Distributed together with: Trouble on the Horizon: Forecasting the Derailment of Online Conversations as they Develop. Jonathan P. Chang and Crisitan Danescu-Niculescu-Mizil. EMNLP 2019.

Dataset details
---------------

User-level information
^^^^^^^^^^^^^^^^^^^^^^

Users in this dataset are Reddit users; their account names are taken as the user names. 

Utterance-level information
^^^^^^^^^^^^^^^^^^^^^^^^^^^

Each utterance corresponds to a Reddit comment. For each utterance, we provide:

* id: Reddit ID of the comment represented by the utterance
* user: the user who author the utterance
* root: Reddit ID of the top-level comment whose thread this utterance belongs to. Note that this differs from how "root" is treated in ConvoKit's general Reddit corpora: in those corpora a conversation is considered to start with a Reddit post, whereas in this corpus a conversation is considered to start with a top-level reply to a post.
* reply_to: Reddit ID of the utterance to which this utterance replies to (None if the utterance represents a top-level comment, i.e., a reply to a post)
* timestamp: time of the utterance
* text: textual content of the utterance

Metadata for each utterance is inherited from the general CMV corpus:

* score: score (i.e., the number of upvotes minus the number of downvotes) of the content 
* top_level_comment: the id of the top level comment (None if the utterance is a post)
* retrieved_on: unix timestamp of the time of when the data is retrieved 
* gilded: gilded status of the content
* gildings: gilding information of the content
* stickied: stickied status of the content
* permalink: permanent link of the content
* author_flair_text: flair of the author 


Conversational-level information
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Metadata for each conversation include:

* pair_id: the conversation ID (root) of the conversation that this conversation is paired with
* has_removed_comment: whether the final comment in this thread was removed by CMV moderators for violation of Rule 2
* split: which split (train, val, or test) this conversation was used in for the experiments described in "Trouble on the Horizon"


Usage
-----

To download directly with ConvoKit: 

>>> from convokit import Corpus, download
>>> corpus = Corpus(filename=download("conversations-gone-awry-cmv-corpus"))


For some quick stats:

>>> len(corpus.get_utterance_ids()) 
42964
>>> len(corpus.get_usernames())
9548
>>> len(corpus.get_conversation_ids())
6842


Contact
^^^^^^^

Please email any questions to: cristian@cs.cornell.edu (Cristian Danescu-Niculescu-Mizil)







