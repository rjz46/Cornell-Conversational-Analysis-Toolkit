from convokit.model import Corpus, User
from convokit.transformer import Transformer
from convokit.toxicity.tox_dictionary import toxicity_scores
from tqdm import tqdm_notebook as tqdm
import random

class Sequence(Transformer):
    """
    Finds a random sequence in a conversation that has more than three comments, annotate 
    the sequence by discourse actions, user after toxicity transformer. The transformer is
    used to randomly sample discourse sequences and get their corresponding toxicity scores. 
    This prevents overlap when sampling sequences in a tree structure.

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

                chosen_chain_tox = []
                chosen_chain_tox.append(utt.meta['toxicity'])

                while(utt.meta['post_depth'] > 0):
                    if utt.reply_to in corpus.utterances:
                        utt = convo.get_utterance(utt.reply_to)
                        chosen_chain.append(utt.id)
                        chosen_chain_tox.append(utt.meta['toxicity'])
                    else:
                        break
                        
                #counter+=1
                chosen_chain.reverse()
                chosen_chain_tox.reverse()
                convo.add_meta('chain', chosen_chain)
                convo.add_meta('chain_tox', chosen_chain_tox)
            else:
                convo.add_meta('chain', None)

        return corpus