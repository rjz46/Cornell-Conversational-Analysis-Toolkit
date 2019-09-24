from convokit.model import Corpus, User
from convokit.transformer import Transformer
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


    def __init__(self, api_key: str=None, toxicity_json_path: str=None, toxicity_path_to_save: str=None):
        if api_key:
            print("WARNING: api key is not currently used; uses hardcoded key.")
        if api_key and toxicity_json_path:
            raise RuntimeError("Nonempty toxicity_json_path was passed, but will be ignored because api_key was set.")
        if not (api_key or toxicity_json_path):
            raise RuntimeError("""Must pass either api_key or toxicity_json_path; \
            if you were relying on the tox_dictionary import, now use \
            'toxicity_json_path=\"convokit/toxicity/data/reddit_coarse_discourse.json\"'""")
        self.api_key = api_key
        self.toxicity_json_path = toxicity_json_path
        self.toxicity_path_to_save = toxicity_path_to_save
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

        if self.toxicity_json_path:
            with open(self.toxicity_json_path, 'r') as f:
                toxicity_scores_dict = json.load(f)
        else:
            toxicity_scores_dict = toxicity_scores


        scores_to_save = {}
        for convo in tqdm(list(corpus.iter_conversations())):

            convo_scores = 0
            count = 0
            for utt in convo.iter_utterances():
                '''
                    rerunning this takes over a day for our 110k+ comments since it uses an api with limited query rate,
                    we'll load them from tox_dictionary.py that was pre-fetched,
                    for others using our transformer, please run self._get_toxicity over the utterances on their corpus.
                '''
                if self.api_key:
                    utt_score = self._get_toxicity(utt.text)
                    scores_to_save[utt.id] = utt_score
                else:
                    utt_score = toxicity_scores_dict[utt.id]

                convo_scores+=utt_score
                count+=1

                #print (utt_score)
                utt.add_meta('toxicity', utt_score)

            convo.add_meta('averagetoxicity', convo_scores/count)

        if self.toxicity_path_to_save:
            with open(self.toxicity_path_to_save, 'w') as f:
                json.dump(scores_to_save, f)

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
