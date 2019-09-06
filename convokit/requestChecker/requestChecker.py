from typing import Callable, Generator, Tuple, List, Dict, Set, Optional, Hashable

from collections import defaultdict

from convokit.politeness_api.features.politeness_strategies import check_elems_for_strategy, initial_polar, aux_polar

from convokit.transformer import Transformer
from convokit.model import Corpus
from tqdm import tqdm

class RequestChecker(Transformer):
    """
    Request checking for utterances in a corpus 
    """

    def __init__(self, verbose: bool=False):
        self.ATTR_NAME = "is_request"
        self.verbose = verbose

    def transform(self, corpus: Corpus):

        """ Check each utterances in the corpus and indicate whether it's likely to be an request.  
        Requires that the corpus has previously been transformed by a Parser,
        such that each utterance has dependency parse info in its metadata table.
        
        :param corpus: the corpus to compute features for.
        :type corpus: Corpus
        """

        # preprocess the utterances in the format expected by the API
        if self.verbose: print("Preprocessing comments...")
        comment_ids, processed_comments = self._preprocess_utterances(corpus)

        # use the bundled politeness API to extract politeness features for each
        # preprocessed comment
        if self.verbose: print("Request checking...")
        labels = [self._check_is_request(doc) for doc in tqdm(processed_comments)]

        # ensure no text is missed
        assert(len(labels) == len(comment_ids))

        # add the extracted strategies to the utterance metadata
        for utt_id, label in zip(comment_ids, labels):
            corpus.get_utterance(utt_id).meta[self.ATTR_NAME] = label

        return corpus

    # borrowed from https://github.com/sudhof/politeness/blob/master/request_utils.py
    def _check_is_request(self, document: List[str]) -> List[bool]:
        """
        Heuristic to determine whether a document
        looks like a request
        :param document- pre-processed document
            that might be a request
        :type document- dict with fields 
            'sentences' and 'parses', as
            in other parts of the system
        """
        for sentence, parse in zip(document['sentences'], document['parses']):
            if "?" in sentence:
                return True
            if check_elems_for_strategy(parse, initial_polar) or check_elems_for_strategy(parse, aux_polar):
                return True
        return False

    def _preprocess_utterances(self, corpus: Corpus) -> Tuple[List[Hashable], List[Dict]]:
        
        """
        Convert each Utterance in the given Corpus into the representation expected
        by the politeness API. Assumes that the Corpus has already been parsed, so that
        each Utterance contains the `parsed` metadata entry
        
        :param corpus: the corpus to compute features for.
        :type corpus: Corpus
        """

        utt_ids = [] # keep track of the order in which we process the utterances, so we can join with the corpus at the end
        documents = []
        for i, utterance in enumerate(tqdm(corpus.iter_utterances())):
            if self.verbose and i > 0 and (i % self.verbose) == 0:
                print("\t%03d" % i)
            utt_ids.append(utterance.id)
            doc = {"text": utterance.text, "sentences": [], "parses": []}
            # the politeness API goes sentence-by-sentence
            
            for sent in utterance.meta["parsed"].sents:
                doc["sentences"].append(sent.text)
                sent_parses = []
                pos = sent.start
                for tok in sent:
                    if tok.dep_ != "punct": # the politeness API does not know how to handle punctuation in parses
                        ele = "%s(%s-%d, %s-%d)"%(tok.dep_, tok.head.text, tok.head.i + 1 - pos, tok.text, tok.i + 1 - pos)
                        sent_parses.append(ele)
                doc["parses"].append(sent_parses)
            
            documents.append(doc)
        
        if self.verbose:
            print("Done!")
        return utt_ids, documents

