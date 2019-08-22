Cornell Movie-Dialogs Corpus
============================

A large metadata-rich collection of fictional conversations extracted from raw movie scripts. (220,579 conversational exchanges between 10,292 pairs of movie characters in 617 movies). 


Distributed together with:
`Chameleons in Imagined Conversations: A new Approach to Understanding Coordination of Linguistic Style in Dialogs <https://www.cs.cornell.edu/~cristian/Chameleons_in_imagined_conversations.html>`_. Cristian Danescu-Niculescu-Mizil and Lillian Lee. Cognitive Modeling and Computational Linguistics Workshop at ACL 2011.

Dataset details
---------------

User-level information
^^^^^^^^^^^^^^^^^^^^^^

Users in this dataset are movie characters. We take user index from the original data release as the user name. For each character, we further provide the following information as user-level metadata:

* character_name: name of the character in the movie
* movie_idx: index of the movie this character appears in
* movie_name: title of the movie
* gender: gender of the character ("?" for unlabeled cases)
* credit_pos: position on movie credits ("?" for unlabeled cases)

Utterance-level information
^^^^^^^^^^^^^^^^^^^^^^^^^^^

For each utterance, we provide:

* id: index of the utterance
* user: the user who author the utterance
* root: index of the conversation root of the utterance
* reply_to: index of the utterance to which this utterance replies to (None if the utterance is not a reply)
* timestamp: time of the utterance
* text: textual content of the utterance

Metadata for utterances include:

* movie_idx: index of the movie from which this utterance occurs
* parsed: parsed version of the utterance text, represented as a SpaCy Doc

Conversational-level information
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Conversations are indexed by the id of the first utterance that make the conversation. 


Corpus-level information
^^^^^^^^^^^^^^^^^^^^^^^^

Additional information for the movies these conversations occur are included as Corpus-level metadata, which includes, for each movie:

* title: movie title
* url: url from which the raw sources were retrieved
* release_year: year of movie release
* rating: IMDB rating of the movie
* votes: number of IMDB votes
* genre: a list of genres this movie belongs to 


Usage
-----

To download directly with ConvoKit:

>>> from convokit import Corpus, download
>>> corpus = Corpus(filename=download("movie-corpus"))


For some quick stats:

>>> len(corpus.get_utterance_ids()) 
304713
>>> len(corpus.get_usernames())
9035
>>> len(corpus.get_conversation_ids())
83097


Additional note
---------------

The original dataset can be downloaded `here <https://www.cs.cornell.edu/~cristian/Chameleons_in_imagined_conversations.html>`_. Refer to the original README for more explanations on dataset construction. 

Contact
^^^^^^^

Please email any questions to: cristian@cs.cornell.edu (Cristian Danescu-Niculescu-Mizil).
