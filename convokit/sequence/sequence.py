from convokit.model import Corpus, User
from convokit.transformer import Transformer
from convokit.toxicity.tox_dictionary import toxicity_scores
from tqdm import tqdm_notebook as tqdm
import random

class Sequence(Transformer):
    """
    Performs toxicity classification on each utterance of a conversation based on
    the text of the conversation. It uses Google's Perspective API for the classification.
    The text are preprocessed to remove any extraneous symbols or characters that might break
    the API call. The text is then sent to the API and retrieves a toxicity score.
    The toxicity score is stored in the utterance level metadata.

    The values returned for the score is in the range 0-1
        0 - 1 (lowest toxicity - highest toxicity)
                
    In addition to the utterance-level toxicity score, the average toxicity score for each
    conversation is also computed and stored in the conversation-level metadata.
    """


    def __init__(self):
        pass

    def transform(self, corpus: Corpus) -> Corpus:
        """Modify the provided corpus. This is an abstract method that must be
        implemented by any Transformer subclass

        :param corpus: the Corpus to transform

        :return: modified version of the input Corpus. Note that unlike the
            scikit-learn equivalent, ``transform()`` operates inplace on the Corpus
            (though for convenience and compatibility with scikit-learn, it also
            returns the modified Corpus).
        """

        #counter = 0
        for convo in corpus.iter_conversations():
            
            temp_chain = []

            for utt in convo.iter_utterances():
                
                if utt.meta['post_depth'] == 2:
                    temp_chain.append(utt.id)
            
            if len(temp_chain) > 0:
                
                convo.add_meta('chain', random.choice(temp_chain))
                
                uttid = random.choice(temp_chain)
                chosen_chain= []
                chosen_chain.append(uttid)

                utt = convo.get_utterance(uttid)
                while(utt.meta['post_depth'] > 0):
                    if utt.reply_to in corpus.utterances:
                        utt = convo.get_utterance(utt.reply_to)
                        chosen_chain.append(utt.id)
                    else:
                        break
                        
                #counter+=1
                chosen_chain.reverse()
                convo.add_meta('chain', chosen_chain)
            else:
                convo.add_meta('chain', None)

        return corpus