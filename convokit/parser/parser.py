import spacy
import sys

from convokit.transformer import Transformer
from convokit.model import Corpus


def _remove_tensor(doc):
    """minimize memory usage of spacy docs by removing the unneeded `tensor` field"""
    doc.tensor = None
    return doc

class Parser(Transformer):
    """
    Transformer that adds SpaCy parses to each Utterance in a Corpus. This 
    parsing step is a prerequisite for most of the models included in convokit.

    :param spacy_nlp: if provided, the Parser will use this SpaCy object to do
        parsing. Otherwise, it will initialize a SpaCy object via load('en')
    """

    def __init__(self, spacy_nlp=None, n_threads=1):
        self.ATTR_NAME = "parsed"
        self.n_threads = n_threads
        if spacy_nlp is not None:
            self.spacy_nlp = spacy_nlp
        else:
            try:
                # no custom spacy object was provided; initialize a generic one based on the
                # default English model. We don't use named entity recognition so we disable
                # that pipeline component for speed purposes.
                self.spacy_nlp = spacy.load('en', disable=['ner'])
            except OSError:
                print("Convokit requires a SpaCy English model to be installed. Run `python -m spacy download en` and retry.")
                sys.exit()

    def transform(self, corpus: Corpus) -> Corpus:
        """Runs the SpaCy parser on each utterance in the corpus, and adds the 
        parses to the utterance metadata table.

        :return: corpus, modified with parses assigned to each utterance
        """
        utt_ids = corpus.get_utterance_ids()
        # if the user specifies multithreading, we will enable parallelized parsing
        # using spacy.pipe. Otherwise we will operate sequentially.
        if self.n_threads == 1:
            spacy_iter = (self.spacy_nlp(corpus.get_utterance(utt_id).text) for utt_id in utt_ids)
        else:
            spacy_iter = self.spacy_nlp.pipe((corpus.get_utterance(utt_id).text for utt_id in utt_ids), n_threads=self.n_threads)
        # add the spacy parses to the utterance metadata
        for utt_id, parsed in zip(utt_ids, spacy_iter):
            corpus.get_utterance(utt_id).meta[self.ATTR_NAME] = _remove_tensor(parsed)
        return corpus