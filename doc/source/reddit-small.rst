Reddit Corpus (small)
=====================
 
A sample of conversations from Reddit from 100 highly active subreddits. From each of these subreddits, we include 100 comments threads that has at least 10 comments each during September, 2018. The complete list of subreddits included can be found `here <https://zissou.infosci.cornell.edu/convokit/datasets/subreddit-corpus/subreddits_small_sample.txt>`_. 


Dataset details
---------------

User-level information
^^^^^^^^^^^^^^^^^^^^^^

Users in this corpus are Reddit users, identified by their account names. The corpus includes the following activity statistics:

* num_posts: number of posts from the user
* num_comments: number of comments from the user


Utterance-level information
^^^^^^^^^^^^^^^^^^^^^^^^^^^

Each individual post or comment is viewed as an utterance. For each utterance, we provide:

* id: index of the utterance
* user: the user who author the utterance
* root: index of the conversation root of the utterance (i.e., the index of the post the utterance belongs to)
* reply_to: index of the utterance to which this utterance replies to (None if the utterance is not a reply)
* timestamp: time of the utterance
* text: textual content of the utterance

Metadata for each utterance may include: 

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

Each post with its corresponding comments are considered a conversation. For each conversation, we provide:

* title: title of the post
* num_comments: number of comments following this post
* domain: domain of the post
* subreddit: subreddit this post is retrieved from
* gilded: gilded status of the post
* gildings: gilding information of the post
* stickied: stickied status of the post
* author_flair_text: flair of the author 


Corpus-level information
^^^^^^^^^^^^^^^^^^^^^^^^

* subreddit: the list of subreddits included in this corpus 
* num_posts: total number of posts included in this corpus
* num_comments: total number of comments in this corpus
* num_user: number of unique users in this corpus


Usage
-----

To download directly with ConvoKit: 

>>> from convokit import Corpus, download
>>> corpus = Corpus(filename=download("reddit-corpus-small"))

For some quick stats:

>>> len(corpus.get_utterance_ids()) 
297132
>>> len(corpus.get_usernames())
119889
>>> len(corpus.get_conversation_ids())
8286

Additional note
---------------

Refer to :doc:`tutorial` for a quick tutorial with ConvoKit with this corpus. 
