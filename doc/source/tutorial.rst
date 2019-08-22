====================
Quick-start tutorial
====================

Setup
=====
Read the `introduction to Convokit <https://convokit.cornell.edu>`_ and the description of its :doc:`architecture </architecture>`.

This toolkit requires Python >=3.6.

If you haven't already,

#. Download the toolkit: ``pip3 install convokit``

#. Download Spacy's English model: ``python3 -m spacy download en``

#. Download nltk's punkt tokenizer: ``import nltk; nltk.download('punkt')`` (in a ``python`` interactive session)

Interactive tutorial
====================
Let us start an interactive session (e.g. with ``python`` or ``ipython``) and import Convokit.

>>> import convokit

Now we load an existing corpus, specifically: `reddit-corpus-small`.

By design, it includes 100 comment threads (each consisting of at least 10 Utterances) from 100 popular subreddits from September 2018.

>>> corpus = convokit.Corpus(filename=convokit.download("reddit-corpus-small"))

Alternatively, if you would like to use a custom corpus, refer to our explanation of the Corpus :doc:`data format </data_format>`.

Exploring the corpus
--------------------

We can examine the corpus metadata:

>>> corpus.meta
{'num_comments': 288846,
 'num_posts': 8286,
 'num_user': 119889,
 'subreddit': 'reddit-corpus-small'}

So the corpus includes 288846 comments and 8286 posts. This is a total of 297132 Utterances. (An Utterance is either a post or a comment in a reddit corpus.)

These 297132 Utterances were made by 119889 different users.

We can get iterators of Utterances, Users, and Conversations, and confirm their sizes match the metadata.

>>> len(list(corpus.iter_users()))
119889
>>> len(list(corpus.iter_utterances()))
297132
>>> len(list(corpus.iter_conversations()))
8286

We can also get a list of Utterance ids.

>>> utter_ids = corpus.get_utterance_ids()

Let's confirm that there are 297132 Utterances as expected.

>>> len(utter_ids)
297132

Let's take the first Utterance id and examine the Utterance it corresponds to:

>>> utter_ids[0]
'9c716m'
>>> corpus.get_utterance(utter_ids[0])
Utterance({'id': '9c716m', 'user': User([('name', 'AutoModerator')]), 'root': '9c716m', 'reply_to': None, 'timestamp': 1535839576, 'text': 'Talk about your day. Anything goes, but subreddit rules still apply. Please be polite to each other! \n', 'meta': {'score': 13, 'top_level_comment': None, 'retrieved_on': 1540061887, 'gilded': 0, 'gildings': {'gid_1': 0, 'gid_2': 0, 'gid_3': 0}, 'subreddit': 'singapore', 'stickied': False, 'permalink': '/r/singapore/comments/9c716m/rsingapore_random_discussion_and_small_questions/', 'author_flair_text': ''}})

Let's explore the Utterance object further.

>>> utt = corpus.get_utterance(utter_ids[0])
>>> utt.meta # Utterance-level metadata
{'score': 13,
 'top_level_comment': None,
 'retrieved_on': 1540061887,
 'gilded': 0,
 'gildings': {'gid_1': 0, 'gid_2': 0, 'gid_3': 0},
 'subreddit': 'singapore',
 'stickied': False,
 'permalink': '/r/singapore/comments/9c716m/rsingapore_random_discussion_and_small_questions/',
 'author_flair_text': ''}
>>> utt.timestamp # the unix timestamp for when the utterance was posted
1535839576
>>> utt.user # the User who posted the Utterance
User([('name', 'AutoModerator')])
>>> utt.user.meta # User-level metadata
{'num_posts': 200, 'num_comments': 27}

Applying a transformer
----------------------

We initialize a HyperConvo transformer, which extracts structural features of conversations through a hypergraph representation.

For HyperConvo specifically, these features are saved to their corresponding conversation's metadata. Other transformers may update User, Utterance, or Corpus metadata instead.

>>> # Limit hypergraph representation to threads of length at least 10,
>>> # using the first 10 utterances
>>> # include_root is set to False as we only want comment threads (i.e. threads that begin
>>> # with the top level comment, not the original post.)
>>> hc = convokit.HyperConvo(prefix_len=10, min_thread_len=10, include_root=False)
>>> hc.fit_transform(corpus)
>>> convos = corpus.iter_conversations()
>>> convo1 = next(iter(convos))
>>> convo1.meta.keys()
dict_keys(['title', 'num_comments', 'domain', 'timestamp', 'subreddit', 'gilded', 'gildings', 'stickied', 'author_flair_text', 'hyperconvo'])
>>> convo1.meta['hyperconvo'] # warning: outputs a lot of text
{'e594ur8': {'max[outdegree over c->c responses]': 1,
  'max[indegree over c->c responses]': 5,
  'argmax[outdegree over c->c responses]': 1,
  'argmax[indegree over c->c responses]': 0,
  'norm.max[outdegree over c->c responses]': 0.1111111111111111,
  'norm.max[indegree over c->c responses]': 0.5555555555555556,
  '2nd-largest[outdegree over c->c responses]': 1,
  '2nd-largest[indegree over c->c responses]': 2,
......

Other transformers can be applied in the same way, and even chained in sequence, as described in the :doc:`Core Concepts Tutorial </architecture>`.

Additional notes
----------------

1. Some corpora are particularly large and may not be initializable in their entirety without significant computational resources. However, it is possible to `partially load utterances from a dataset <https://github.com/CornellNLP/Cornell-Conversational-Analysis-Toolkit/blob/master/tests/test_corpus_partial_load.ipynb>`_ to carry out processing of large corpora sequentially.

2. It is possible to `merge two different Corpora (even when there are overlaps or conflicts in Corpus data) <https://github.com/CornellNLP/Cornell-Conversational-Analysis-Toolkit/blob/master/examples/merging/corpus_merge_demo.ipynb>`_

3. See :doc:`examples` for more illustrations of Corpus and Transformer functionality.



