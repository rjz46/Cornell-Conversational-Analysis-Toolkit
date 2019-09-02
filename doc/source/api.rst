API Reference
=============

Core Classes
------------

These are the classes described in :doc:`Core Concepts Tutorial </architecture>`

.. toctree::
   :maxdepth: 2

   User <user.rst>
   Utterance <utterance.rst>
   Conversation <conversation.rst>
   Corpus <corpus.rst>
   Transformer <transformer.rst>

Utilities
---------

Miscellaneous utility functions for managing datasets

.. toctree::
   :maxdepth: 2

   Util <util.rst>

Conversational Analysis Modules
-------------------------------

These are the main offerings of ConvoKit: standalone modules implementing various functions for analyzing conversational behavior. Each module is a Transformer subclass.

.. toctree::
   :maxdepth: 2

   Parser <parser.rst>
   Coordination <coordination.rst>
   QuestionTypology <questionTypology.rst>
   PolitenessStrategies <politenessStrategies.rst>
   Hyperconvo <hyperconvo.rst>
   ThreadEmbedder <threadEmbedder.rst>
   CommunityEmbedder <communityEmbedder.rst>