Coarse Discourse Sequence Corpus
=========================================

Data from paper `Characterizing Online Discussion Using Coarse Discourse Sequences <https://ai.google/research/pubs/pub46055>`_

Coarse Discourse, the Reddit dataset that contains ~9K threads, with comments annotated with 9 main labels and an “other” label:

* Question & Request
* Answer
* Announcement
* Agreement
* Appreciation & Positive Reaction
* Disagreement
* Negative Reaction
* Elaboration & FYI
* Humor
* Other

Dataset details
---------------

User-level information
^^^^^^^^^^^^^^^^^^^^^^

Our users are Reddit users, with their name being their Reddit username. Users who deleted their accounts have their name listed as ‘[deleted]’. 

Each user has the following:

* name: Reddit username
* utts: {‘post_id’: convokit.Utterance} - all the utterances of the user
* convos : {‘convo_id’: convokit.Conversation} - all the conversations user appears in

The users are first initialized with only the usernames, and then their list of utterances and conversations is added in after the utterances and then the conversations are built. 

Utterance-level information
^^^^^^^^^^^^^^^^^^^^^^^^^^^

Each utterance has an id based on the Reddit comment or post id. 
The utterance also contains 

* id: unique_id of the utterance
* user: author of the post/comment as an object
* root: root id/post id that the comment belongs to
* in_reply_to: the comment/post that it replies to
* text:  textual content of the utterance, none if there is no body in the text


Additional information including the annotations for discourse actions that are specific to this dataset and the information specific to reddit are contained in the meta data: 

* post_depth: depth of the post, 0 if post depth of the utterance is the post itself
* majority type: discourse action type by one of the following: question, answer, announcement, agreement,  appreciation, disagreement, elaboration, humor
* ann_types (list of annotation types by three annotators)
* majority_link : link in relation to previous post, none if no relation with previous comment
* ann_links (list of annotation links by three annotators)
* ups : number of votes (upvotes - downvotes) for the comment/post 
    

Conversational-level information
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Each conversation has the following:

* Owner: Reddit username of original poster
* Id: original post Reddit ID
* Utterances: list of Reddit post/comment ids associated with the conversation
* meta_data = {
        "subreddit": associated subreddit, 
        "url": original post url,
        "title": original post title
  }

The metadata include the subreddit, original post URL, and the title of the original post.

Corpus-level information
^^^^^^^^^^^^^^^^^^^^^^^^

We didn’t include any additional corpus metadata.


Some stats on the data set:

>>> len(corpus.get_utterance_ids()) 
115827
>>> len(corpus.get_usernames())
63573
>>> len(corpus.get_conversation_ids())
9483


Contact
^^^^^^^
Ru Zhao, Katy Blumer, Andrew Semmes

Please email any questions to: {rjz46, ,als452} @cornell.edu



