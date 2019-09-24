from convokit.model import Corpus, User
from convokit.transformer import Transformer
from convokit.toxicity.tox_dictionary import toxicity_scores
import requests
import json
import time
from tqdm import tqdm_notebook as tqdm


class Toxicity(Transformer):
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

    @staticmethod
    def _get_toxicity(text):

        #headers and parameters for perspective api call
        headers = {
            'Content-Type': 'application/json',
        }

        params = [
            (('key', 'AIzaSyDyRDMXjs3UFWxmsAcyBnkTG5dLgK4Jjzw'), ) 
        ]

        text = text.encode('utf-8')
        
        line = ''
        for a in text:
            a = chr(a)
            if a=='[':
                f=False
                break
            if a==' ' or (a<='Z' and a>='A') or (a<='z' and a>='a') or (a<='9' and a>='0') or a=='?' or a=='.':
                line +=a


        if len(line) > 0:
            try:
                data = '{comment: {text:"'+line+'"}, languages: ["en"], requestedAttributes: {TOXICITY:{}} }'
                response = requests.post('https://commentanalyzer.googleapis.com/v1alpha1/comments:analyze', headers=headers, params=params[0], data=data)
                j = json.loads(response.text)
                return j['attributeScores']['TOXICITY']['summaryScore']['value']
            except:
                print("ERROR1!!!!!!!!!!!!!!!!!!!!")
                try:
                    time.sleep(2)
                    data = '{comment: {text:"'+line+'"}, languages: ["en"], requestedAttributes: {TOXICITY:{}} }'
                    response = requests.post('https://commentanalyzer.googleapis.com/v1alpha1/comments:analyze', headers=headers, params=params[0], data=data)
                    j = json.loads(response.text)

                    return j['attributeScores']['TOXICITY']['summaryScore']['value']
                except:
                    print("ERROR2!!!!!!!!!!!!!!!!!!!!")

                    try:
                        time.sleep(2)
                        data = '{comment: {text:"'+line+'"}, languages: ["en"], requestedAttributes: {TOXICITY:{}} }'
                        response = requests.post('https://commentanalyzer.googleapis.com/v1alpha1/comments:analyze', headers=headers, params=params[0], data=data)
                        j = json.loads(response.text)

                        return j['attributeScores']['TOXICITY']['summaryScore']['value']
                    except:
                        print("ERROR3")
                        print(j)
        return 0.0

    def transform(self, corpus: Corpus) -> Corpus:
        """Modify the provided corpus. This is an abstract method that must be
        implemented by any Transformer subclass

        :param corpus: the Corpus to transform

        :return: modified version of the input Corpus. Note that unlike the
            scikit-learn equivalent, ``transform()`` operates inplace on the Corpus
            (though for convenience and compatibility with scikit-learn, it also
            returns the modified Corpus).
        """

        for convo in tqdm(list(corpus.iter_conversations())):

            convo_scores = 0
            count = 0

            for utt in convo.iter_utterances():

                '''
                    rerunning this takes over a day for our 110k+ comments since it uses an api with limited query rate,
                    we'll load them from tox_dictionary.py that was pre-fetched,
                    for others using our transformer, please run self._get_toxicity over the utterances on their corpus.
                '''

                #utt_score = self._get_toxicity(utt.text)

                utt_score = toxicity_scores[utt.id]

                convo_scores+=utt_score
                count+=1

                #print (utt_score)
                utt.add_meta('toxicity', utt_score)

            convo.add_meta('averagetoxicity', convo_scores/count)

        return corpus

    def fit(self, corpus: Corpus):
        """Use the provided Corpus to perform any precomputations necessary to
        later perform the actual transformation step.

        :param corpus: the Corpus to use for fitting

        :return: the fitted Transformer
        """


        return self

    def fit_transform(self, corpus: Corpus) -> Corpus:
        """Fit and run the Transformer on a single Corpus.

        :param corpus: the Corpus to use

        :return: same as transform
        """
        self.fit(corpus)
        return self.transform(corpus)
