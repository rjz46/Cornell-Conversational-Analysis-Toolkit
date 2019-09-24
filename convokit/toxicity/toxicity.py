from convokit.model import Corpus, User
from convokit.transformer import Transformer
from tox_dictionary import toxicity_scores

class Toxicity(Transformer):
    """
    Abstract base class for modules that take in a Corpus and modify the Corpus
    and/or extend it with additional information, imitating the scikit-learn
    Transformer API. Exposes ``fit()`` and ``transform()`` methods. ``fit()`` performs any
    necessary precomputation (or “training” in machine learning parlance) while
    ``transform()`` does the work of actually computing the modification and
    applying it to the corpus. 

    All subclasses must implement ``transform()``;
    subclasses that require precomputation should also override ``fit()``, which by
    default does nothing. Additionally, the interface also exposes a
    ``fit_transform()`` method that does both steps on the same Corpus in one line.
    By default this is implemented to simply call ``fit()`` followed by ``transform()``,
    but designers of Transformer subclasses may also choose to overwrite the
    default implementation in cases where the combined operation can be
    implemented more efficiently than doing the steps separately.
    """

    #headers and parameters for perspective api call
    headers = {
        'Content-Type': 'application/json',
    }

    params = (
        ('key', '[api-key]'),
    )

    def __init__(self):
        pass

    @staticmethod
    def _preprocess(text):
        body_or_title = text.encode('utf-8')        
        result = ''
        for a in body_or_title: 
            a = chr(a)
            if a=='[':
                f=False
                break
            if a==' ' or (a<='Z' and a>='A') or (a<='z' and a>='a') or (a<='9' and a>='0') or a=='?' or a=='.':
                result +=a
        return result

    @staticmethod
    def _get_toxicity(self, line):
    
        line = self._preprocess(line)
        
        global get_toxicity_count

        if len(line) > 0:
            try:
                data = '{comment: {text:"'+line+'"}, languages: ["en"], requestedAttributes: {TOXICITY:{}} }'
                response = requests.post('https://commentanalyzer.googleapis.com/v1alpha1/comments:analyze', headers=headers, params=params[0], data=data)
                j = json.loads(response.text)
                get_toxicity_count +=1
                return j['attributeScores']['TOXICITY']['summaryScore']['value']
            except:
                print("ERROR1!!!!!!!!!!!!!!!!!!!!" + str(get_toxicity_count))
                try:
                    time.sleep(2)
                    data = '{comment: {text:"'+line+'"}, languages: ["en"], requestedAttributes: {TOXICITY:{}} }'
                    response = requests.post('https://commentanalyzer.googleapis.com/v1alpha1/comments:analyze', headers=headers, params=params[0], data=data)
                    j = json.loads(response.text)
                    
                    return j['attributeScores']['TOXICITY']['summaryScore']['value']
                except:
                    print("ERROR2!!!!!!!!!!!!!!!!!!!!" + str(get_toxicity_count))

                    try:
                        time.sleep(2)
                        data = '{comment: {text:"'+line+'"}, languages: ["en"], requestedAttributes: {TOXICITY:{}} }'
                        response = requests.post('https://commentanalyzer.googleapis.com/v1alpha1/comments:analyze', headers=headers, params=params[0], data=data)
                        j = json.loads(response.text)
                        
                        return j['attributeScores']['TOXICITY']['summaryScore']['value']
                    except:
                
                        print("ERROR3" + str(get_toxicity_count))
                        #blacklist.append(current_target)

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

        for convo in corpus.iter_conversations():
            convo_scores = 0
            count = 0

            for utt in convo.iter_utterances():        
                '''
                    rerunning this takes over a day for our 110k+ comments since it uses an api with limited query rate, 
                    we'll load them from toxicity_dictionary.json that was pre-fetched,
                    for others using our transformer, please run self.get_toxicity over the utterances on their corpus.
                '''

                #utt_score = self._get_toxicity(utt.text)


                utt_score = toxicity_scores[utt.id]
                
                convo_scores+=utt_score
                count+=1

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
