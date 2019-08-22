

import itertools
import json
import numpy as np
import pandas as pd
import spacy
import pickle

from ast import literal_eval as make_tuple
from collections import defaultdict, Counter
from scipy import sparse
from sklearn.cluster import KMeans
from sklearn.preprocessing import Normalizer
from spacy.symbols import *
from typing import Callable, Generator, Tuple, List, Dict, Set, Optional

from convokit.model import Corpus
from convokit.transformer import Transformer

NP_LABELS = {nsubj, nsubjpass, dobj, iobj, pobj, attr}
SPACY_META = "parsed"
pair_delim = '-q-a-'
span_delim = 'span'

class QuestionTypology(Transformer):
    """Encapsulates computation of question types from a question-answer corpus.
    Can be trained and evaluated on separate corpora.

    :param num_clusters: the number of question types to be extracted
    :param question_threshold: the minimum number of questions a motif must occur in for it to be considered
    :param answer_threshold: the minimum number of answers a motif must occur in for it to be considered
    :param num_dims: the number of latent dimensions in the sparse matrix
    :param verbose: False or 0 if nothing should be printed, otherwise equal to the interval at which the number of completed steps of any part of the algorithm are printed
    :param dedup_threshold: If two motifs co-occur in a higher proportion of cases than this threshold, they are considered duplicates and one is removed
    :param follow_conj: whether to follow conjunctions and treat subtrees as sentences too.
    :param norm: the normalizer to use in the normalization of the sparse matrix
    :param num_svds: the number of dimensions to preserve in the SVD
    :param num_dims_to_inspect: the number of dimensions to inspect
    :param max_iter_for_k_means: the maximum iterations to run the k means algorithm for
    :param remove_first: Whether to remove the first element in the k means classification set
    :param min_support: the minimum number of times an itemset has to show up for the frequent itemset counter to consider
    :param item_set_size: the size of the item set
    :param leaves_only_for_assign: whether to assign only sink motifs to clusters
    :param idf: Whether to represent data using inverse document frequency
    :param snip: Whether to increment the number of singular values and vectors to compute by one
    :param leaves_only_for_extract: whether to include only sink motifs in extracted clusters
    :param random_seed: the random seed to provide to the clustering algorithm
    :param is_question: the function that will be used to determine whether an utterance is a question.
                        If nothing is specified, by default the code assumes all sentences that
                        end in '?' are questions
    :param questions_only: whether motif extraction should look only at utterances that
                           are questions (as defined by is_question). Disable this
                           to make the algorithm derive prompt types instead of
                           question types.
    :param enforce_formatting: whether to enforce that utterances must be well-formed
                               sentences in order to count as questions or answers.
                               Well-formedness is defined as starting with an uppercase
                               letter. Enable this for corpora that are known to contain
                               properly formatted utterances (e.g. Parliament corpus)

    :ivar num_clusters: the number of question types to be extracted
    :ivar mtx_obj: an object that contains information about the QA matrix from the paper
    :ivar km: the Kmeans object that has the labels
    :ivar types_to_data: an object that contains information about motifs, fragments and questions in each type
    :ivar lq: the low dimensional Q matrix
    :ivar a_u: the low dimensional A matrix
    """

    def __init__(self, num_clusters: int=8,
        question_threshold: int=100, answer_threshold: int=100,
        num_dims: int=100, verbose: int=5000, dedup_threshold: int=.9,
        follow_conj: int=True, norm: str='l2', num_svds: int=50, num_dims_to_inspect: int=5,
        max_iter_for_k_means: int=1000, remove_first: bool=False, min_support: int=5, item_set_size: int=5,
        leaves_only_for_assign: bool=True, idf: bool=False, snip: bool=True, leaves_only_for_extract: bool=False,
        random_seed: int=0, is_question: Callable[[str], bool]=None, questions_only: bool=True, enforce_formatting: bool=True):

        self.num_clusters = num_clusters
        self.question_threshold = question_threshold
        self.answer_threshold = answer_threshold
        self.num_dims = num_dims
        self.verbose = verbose
        self.dedup_threshold = dedup_threshold
        self.follow_conj = follow_conj
        self.norm = norm
        self.num_svds = num_svds
        self.num_dims_to_inspect = num_dims_to_inspect
        self.max_iter_for_k_means = max_iter_for_k_means
        self.remove_first = remove_first
        self.min_support = min_support
        self.item_set_size = item_set_size
        self.leaves_only_for_assign = leaves_only_for_assign
        self.idf = idf
        self.snip = snip
        self.leaves_only_for_extract = leaves_only_for_extract
        self.random_seed = random_seed
        if not is_question: is_question = MotifsExtractor.is_utterance_question

        if questions_only:
            self.is_question = is_question
            if enforce_formatting:
                self.question_filter = lambda x: (self.is_question(x) and MotifsExtractor.is_uppercase(x))
            else:
                self.question_filter = self.is_question
        else:
            self.is_question = lambda x: True
            self.question_filter = lambda x: True
        if enforce_formatting:
            self.answer_filter = MotifsExtractor.is_uppercase
        else:
            self.answer_filter = lambda x: True

    def fit(self, corpus: Corpus):
        """Extract question-answer pairs from the given corpus and use them to
        construct the internal matrix objects (in other words, "train" the
        QuestionTypology object on the given corpus)
        
        :param corpus: the Corpus to use for fitting the model
        :type corpus: Corpus
        """
        self.motifs = MotifsExtractor.extract_question_motifs(QuestionTypology._iter_corpus(corpus, 'questions', self.is_question),
            self.question_filter, self.follow_conj, self.min_support, self.dedup_threshold, self.item_set_size, self.verbose)
        self.motifs["answer_arcs"] = MotifsExtractor.extract_answer_arcs(QuestionTypology._iter_corpus(corpus, 'answers', self.is_question),
            self.answer_filter, self.follow_conj, self.verbose)

        self.mtx_obj = QuestionClusterer.build_matrix(self.motifs, self.question_threshold,
            self.answer_threshold, self.verbose)

        self.km, self.types_to_data, self.lq, self.a_u, self.a_s, self.a_v = \
        QuestionClusterer.extract_clusters(self.mtx_obj,
            self.num_clusters,self.num_dims, self.snip, self.verbose, self.norm,
            self.idf, self.leaves_only_for_extract, self.remove_first, self.max_iter_for_k_means,
            self.random_seed)

        self.motif_df, self.aarc_df, self.qdoc_df, self.q_leaves, self.qdoc_vects = QuestionClusterer.assign_clusters(self.km,
            self.lq, self.a_u, self.mtx_obj, self.num_dims, self.norm,
            self.idf, self.leaves_only_for_assign)

        for index, row in self.qdoc_df.iterrows():
            cluster = row["cluster"]
            cluster_dist = row["cluster_dist"]
            all_cluster_dists = row["all_cluster_dists"]
            q_idx = QuestionTypologyUtils.get_q_idx_from_pair(row["q_idx"])
            self.types_to_data[cluster]["questions"].append(q_idx)
            self.types_to_data[cluster]["question_dists"].append(cluster_dist)

        self._calculate_totals()

        return self

    @staticmethod
    def _iter_corpus(corpus: Corpus, iter_type: str,
                     is_utterance_question: Callable[[str], bool]) -> Generator[Tuple[str, str, str], None, None]:
        """Iterator over utterances in the Corpus being transformed

        Can give just questions, just answers or questions followed by their answers
        """
        i = -1
        for utterance in corpus.iter_utterances():
            if utterance.reply_to is not None:
                root_text = corpus.get_utterance(utterance.reply_to).text
                if is_utterance_question(root_text):
                    i += 1
                    if iter_type == 'answers':
                        pair_idx = utterance.reply_to + pair_delim + utterance.id
                        yield utterance.id, utterance.meta[SPACY_META], pair_idx
                        continue
                    question = corpus.get_utterance(utterance.reply_to)
                    pair_idx = question.id + pair_delim + utterance.id
                    yield question.id, question.meta[SPACY_META], pair_idx
                    if iter_type == 'both':
                        pair_idx = utterance.reply_to + pair_delim + utterance.id
                        yield utterance.id, utterance.meta[SPACY_META], pair_idx

    def _calculate_totals(self):
        """Calculates variables for display. Calculates total questions, total extracted motifs,
            total answer fragments, total number of motifs in each cluster, and the total nunber
            of questions assigned to each question type.
        """

        self.num_questions = 0
        self.num_motifs = 0
        self.num_fragments = 0
        self.motifs_in_each_cluster = [0 for i in range(self.num_clusters)]
        self.questions_in_each_cluster = [0 for i in range(self.num_clusters)]

        for cluster in self.types_to_data.keys():
            self.num_questions += len(self.types_to_data[cluster]['questions'])
            self.num_motifs += len(self.types_to_data[cluster]['motifs'])
            self.num_fragments += len(self.types_to_data[cluster]['fragments'])

        for label in self.types_to_data:
            self.questions_in_each_cluster[label] = len(self.types_to_data[label]["questions"])

        for label in self.km.labels_:
            self.motifs_in_each_cluster[label] += 1

    def display_totals(self):
        """Displays the total number of questions, motifs and fragments
            present in this data, as well as the number of motifs in each cluster
            and questions of each type
        """
        print("Total Motifs: %d"%self.num_motifs)
        print("Total Questions: %d"%self.num_questions)
        print("Total Fragments: %d"%self.num_fragments)
        print("Number of Motifs in each cluster: ", self.motifs_in_each_cluster)
        print("Number of Questions of each type: ", self.questions_in_each_cluster)

    @staticmethod
    def display_questions_for_type(corpus: Corpus, type_num: int, num_egs: int=10):
        """Displays num_egs number of questions in the given corpus that were assigned type
            type_num by the typing algorithm.
        """
        questions = []
        question_dists = []
        for utterance in corpus.iter_utterances():
            if "qtype" in utterance.meta and utterance.meta["qtype"] == type_num:
                questions.append(utterance.text)
                question_dists.append(utterance.meta["qtype_dists"][type_num])
        questions_len = len(questions)
        num_to_print = min(questions_len, num_egs)
        indices_to_print = np.argsort(question_dists)[:num_to_print]
        print('\t%d sample questions that were assigned type %d (%d total questions with this type) :'%(num_to_print, type_num, questions_len))
        n = 0
        for i in indices_to_print:
            n += 1
            print('\t\t%d.'%(n), questions[i])

    @staticmethod
    def display_question_answer_pairs_for_type(corpus: Corpus, type_num: int, num_egs: int=10):
        """Displays num_egs number of question-answer pairs in the given corpus that were assigned type
            type_num by the typing algorithm.
        """
        questions = []
        question_dists = []
        answers = []
        for utterance in corpus.iter_utterances():
            if utterance.reply_to is not None:
                question = corpus.get_utterance(utterance.reply_to)
                if "qtype" in question.meta and question.meta["qtype"] == type_num:
                    questions.append(question.text)
                    question_dists.append(question.meta["qtype_dists"][type_num])
                    answers.append(utterance.text)
        questions_len = len(questions)
        num_to_print = min(questions_len, num_egs)
        indices_to_print = np.argsort(question_dists)[:num_to_print]
        print('\t%d sample question-answer pairs that were assigned type %d (%d total questions with this type) :'%(num_to_print, type_num, questions_len))
        n = 0
        for i in indices_to_print:
            n += 1
            print('\t\tQuestion %d.'%(n), questions[i])
            print('\t\tAnswer %d.'%(n), answers[i])

    def display_motifs_for_type(self, cluster_num: int, num_egs: int=10):
        """Displays num_egs number of motifs that were assigned to cluster cluster_num
            by the clustering algorithm
        """
        target = self.types_to_data[cluster_num]
        motifs = target["motifs"]
        motif_dists = target["motif_dists"]
        motifs_len = len(motifs)
        num_to_print = min(motifs_len, num_egs)
        indices_to_print = np.argsort(motif_dists)[:num_to_print]
        print('\t%d sample question motifs for type %d (%d total motifs):'%(num_to_print, cluster_num, motifs_len))
        n = 0
        for i in indices_to_print:
            n += 1
            print('\t\t%d.'%(n), motifs[i])

    def display_answer_fragments_for_type(self, cluster_num, num_egs=10):
        """Displays num_egs number of answer fragments whose corresponding
            question motif were assigned to cluster cluster_num by the clustering algorithm
        """
        target = self.types_to_data[cluster_num]
        answer_fragments = target["fragments"]
        fragment_dists = target["fragment_dists"]
        fragment_len = len(answer_fragments)
        num_to_print = min(fragment_len, num_egs)
        indices_to_print = np.argsort(fragment_dists)[:num_to_print]
        print('\t%d sample answer fragments for type %d (%d total fragments) :'%(num_to_print, cluster_num, fragment_len))
        n = 0
        for i in indices_to_print:
            n += 1
            print('\t\t%d.'% n, answer_fragments[i])

    def _summarize_motifs(self) -> Tuple[List, List]:
        """Helper function to summarize question motifs and corresponding answer
        fragments for inclusion in the transformed corpus"""
        motif_summary = []
        answer_summary = []
        for cl in range(self.num_clusters):
            target = self.types_to_data[cl]
            motifs = target["motifs"]
            motifs_idx = np.argsort(target["motif_dists"])
            motif_summary.append([motifs[i] for i in motifs_idx])
            answer_fragments = target["fragments"]
            fragments_idx = np.argsort(target["fragment_dists"])
            answer_summary.append([answer_fragments[i] for i in fragments_idx])
        return motif_summary, answer_summary

    def _corpus_to_dataframe(self, corpus: Corpus) -> pd.DataFrame:
        comment_ids = []
        content = []
        for utt in corpus.iter_utterances():
            if self.is_question(utt.text):
                comment_ids.append(utt.id)
                content.append(utt.meta["parsed"])
        return pd.DataFrame({"content": content}, index=comment_ids)

    def _load_motif_info(self) -> Tuple[Dict[Tuple, Tuple], Dict[Tuple, List[Tuple]], Dict[Tuple, int]]:
        if self.verbose:
            print("fitting extracted motifs to new data...")

        super_mappings = {}
        for entry in self.motifs['question_supersets_arcset_to_super']:
            super_mappings[tuple(entry['arcset'])] = tuple(entry['super'])

        downlinks = MotifsExtractor.read_downlinks(self.motifs['question_tree_downlinks'])
        node_counts = MotifsExtractor.read_nodecounts(self.motifs['question_tree_arc_set_counts'])
        return super_mappings, downlinks, node_counts

    def _extract_arcs(self, comment_df: pd.DataFrame,
                      selector: Callable[[str], bool]=lambda x: True,
                      outfile: str=None) -> pd.DataFrame:
        if self.verbose:
            print("getting question arcs")
        sent_df = []
        for i, tup in enumerate(comment_df.itertuples()):
            if self.verbose and i > 0 and (i % self.verbose) == 0:
                print("\t%03d" % i)
            for s_idx, sent in enumerate(tup.content.sents):
                sent_text = sent.text.strip()
                if len(sent_text) == 0: continue
                if selector(sent_text):
                    sent_df.append({
                            'idx': tup.Index, 'sent_idx': s_idx, 'span': sent, 
                            'arc_sets': MotifsExtractor.get_arcs(sent.root, True),
                            'content': sent_text, 'sent_key': tup.Index + '_' + str(s_idx)
                        })
        sent_df = pd.DataFrame(sent_df)
        if outfile is not None:
            sent_df.to_csv(outfile + '.sent_arcs.tsv', sep='\t')
        return sent_df

    def _fit_questions_and_answers(self, sent_df: pd.DataFrame, q_vocab: Set[str], a_vocab: Set[str],
                                   super_mappings: Dict[Tuple, Tuple],
                                   downlink_info: Dict[Tuple, List[Tuple]],
                                   node_count_info: Dict[Tuple, int],
                                   threshold: int, outfile: str=None, per_sent: bool=False):

        if self.verbose:
            print("fitting motifs to questions")

        question_to_fits = defaultdict(set)
        question_to_leaf_fits = defaultdict(set)
        question_to_a_fits = defaultdict(set)

        for i, tup in enumerate(sent_df.itertuples()):
            if self.verbose and i > 0 and (i % self.verbose) == 0:
                print("\t%03d" % i)
            if per_sent:
                key = tup.sent_key
            else:
                key = tup.idx
            for arc in tup.arc_sets:
                if arc in a_vocab: question_to_a_fits[key].add(arc)

            motif_fits = MotifsExtractor.fit_question(tup.arc_sets, downlink_info, node_count_info)
            for entry in motif_fits.values():
                motif = entry['arcset']
                if motif == ('*', ): continue
                super_motif = super_mappings.get(motif, '')
                if super_motif not in q_vocab: continue
                if entry['arcset_count'] < threshold: continue
                if entry['max_valid_child_count'] < threshold:
                    question_to_leaf_fits[key].add(super_motif)
                question_to_fits[key].add(super_motif)
        if outfile is not None:
            df = pd.DataFrame.from_dict({
                    'question_fits': question_to_fits,
                    'question_leaf_fits': question_to_leaf_fits,
                    'question_a_fits': question_to_a_fits
                })
            df.to_csv(outfile + '.fits.tsv', sep='\t')
        return question_to_fits, question_to_leaf_fits, question_to_a_fits 

    def _make_new_qa_mtx_obj(self, question_to_fits, question_to_leaf_fits, question_to_a_fits, ref_mtx_obj,
            outfile=None):

        if self.verbose:
            print("building new q-a matrices")

        docs = [x for x,y in question_to_fits.items() if len(y) > 0]
        doc_to_idx = {doc:idx for idx,doc in enumerate(docs)}
        qterm_idxes = []
        leaves = []
        qdoc_idxes = []
        aterm_idxes = []
        adoc_idxes = []

        for i, doc in enumerate(docs):
            if self.verbose and i > 0 and (i % self.verbose) == 0:
                print("\t%03d" % i)
            qterms = question_to_fits[doc]
            for term in qterms:
                qterm_idxes.append(ref_mtx_obj['q_term_to_idx'][term])
                leaves.append(term in question_to_leaf_fits[doc])
                qdoc_idxes.append(doc_to_idx[doc])
            aterms = question_to_a_fits[doc]
            for term in aterms:
                aterm_idxes.append(ref_mtx_obj['a_term_to_idx'][term])
                adoc_idxes.append(doc_to_idx[doc])

        qterm_idxes = np.array(qterm_idxes)
        leaves = np.array(leaves)
        qdoc_idxes = np.array(qdoc_idxes)
        aterm_idxes = np.array(aterm_idxes)
        adoc_idxes = np.array(adoc_idxes)
        new_mtx_obj = {'q_terms': ref_mtx_obj['q_terms'], 'q_didxes': qdoc_idxes, 'docs': docs, 'q_leaves': leaves,
                    'q_term_counts': ref_mtx_obj['q_term_counts'], 'q_term_to_idx': ref_mtx_obj['q_term_to_idx'],
                    'doc_to_idx': doc_to_idx, 'q_tidxes': qterm_idxes, 'N_idf_docs': len(ref_mtx_obj['docs']),
                    'a_terms': ref_mtx_obj['a_terms'],
                    'a_term_counts': ref_mtx_obj['a_term_counts'], 'a_term_to_idx': ref_mtx_obj['a_term_to_idx'],
                    'a_tidxes': aterm_idxes, 'a_didxes': adoc_idxes}
        if outfile is not None:
            np.save(outfile + '.q.tidx.npy', qterm_idxes)
            np.save(outfile + '.q.leaves.npy', leaves)
            np.save(outfile + '.a.tidx.npy', aterm_idxes)
            np.save(outfile + '.q.didx.npy', qdoc_idxes)
            np.save(outfile + '.a.didx.npy', adoc_idxes)
            with open(outfile + '.docs.txt', 'w') as f:
                f.write('\n'.join(docs))

        return new_mtx_obj

    def _project_qa_embeddings(self, mtx_obj, lq, au, outfile=None):

        if self.verbose:
            print("\tbuilding matrices")

        qmtx = QuestionClusterer.build_mtx(mtx_obj,'q',norm='l2', idf=False, leaves_only=True)
        amtx = QuestionClusterer.build_mtx(mtx_obj, 'a', norm='l2', idf=True, leaves_only=False)

        lq_norm = Normalizer().fit_transform(lq)
        au_norm = Normalizer().fit_transform(au)

        qdoc_vects = Normalizer().fit_transform(qmtx.T) * lq_norm
        adoc_vects = ((amtx.T) * au)

        if outfile is not None:
            np.save(outfile + '.qdoc', qdoc_vects)
            np.save(outfile + '.adoc', adoc_vects)

        return qdoc_vects, adoc_vects

    @staticmethod
    def _assign_qtypes(qdoc_vects, adoc_vects, mtx_obj, km, comment_df,
            display=None, max_dist_quantile=None, random_state=None, outfile=None):

        n_clusters = km.n_clusters
        qdoc_norm = Normalizer().fit_transform(qdoc_vects)
        adoc_norm = Normalizer().fit_transform(adoc_vects)

        qdoc_dists = km.transform(qdoc_norm)
        qdoc_df = pd.DataFrame(data=qdoc_dists, index=mtx_obj['docs'], columns=["km_%d_dist" % i for i in range(n_clusters)])
        return qdoc_df

    def transform(self, corpus) -> Corpus:
        """Computes the distance to each question type cluster for some (possibly previously unseen) text.
        
        :param corpus: the Corpus to apply the fitted model to
        :type corpus: Corpus
        """

        if self.verbose:
            print("transforming corpus!")

        # convert corpus utterances to dataframe for easier indexing later
        comment_df = self._corpus_to_dataframe(corpus)

        qvocab = set(self.mtx_obj['q_terms'])
        avocab = set(self.mtx_obj['a_terms'])

        # fit motifs to new data
        super_mappings, downlinks, node_counts = self._load_motif_info()
        sent_df = self._extract_arcs(comment_df)
        question_to_fits, question_to_leaf_fits, question_to_a_fits = self._fit_questions_and_answers(sent_df, qvocab, 
            avocab, super_mappings, downlinks, node_counts, self.question_threshold)

        # project new data
        new_mtx_obj = self._make_new_qa_mtx_obj(question_to_fits, question_to_leaf_fits, question_to_a_fits, self.mtx_obj)
        qdoc_vects, adoc_vects = self._project_qa_embeddings(new_mtx_obj, self.lq, self.a_u)

        new_qdoc_df = QuestionTypology._assign_qtypes(qdoc_vects, adoc_vects, new_mtx_obj, self.km, comment_df,
            random_state=self.random_seed, display=5, max_dist_quantile=.25)

        # add cluster assignments to the source Corpus
        for utt_id in new_qdoc_df.index:
            utterance = corpus.get_utterance(utt_id)
            utterance.meta["qtype"] = np.argmin(new_qdoc_df.loc[utt_id].values)
            utterance.meta["qtype_dists"] = new_qdoc_df.loc[utt_id].values

        # add information for interpreting the question types to the corpus metadata
        motif_summary, answer_summary = self._summarize_motifs()
        corpus.add_meta("motifs", motif_summary)
        corpus.add_meta("answer_fragments", answer_summary)

        if self.verbose:
            print("done!")

        return corpus


class MotifsExtractor:

    @staticmethod
    def load_vocab(verbose):
        """
            Returns a spacy Vocab object for the English vocabulary
        """
        if verbose:
            print('loading spacy vocab')
        return spacy.load('en').vocab

    @staticmethod
    def iterate_spacy(path, vocab):
        with open(path + '.pk', 'rb') as spacy_file:
            objs = pickle.load(spacy_file)
        with open(path + '.txt') as key_file:
            for obj in objs:
                try:
                    key = next(key_file)
                    yield key.strip(), obj
                except Exception as e:
                    print(e)

    @staticmethod
    def get_spacy_dict(path, vocab, verbose):
        """
            gets a dict of (key --> spacy object) from a path (as generated by the spacify function).
            can pass pre-loaded vocabulary to avoid the terrible load time.

            currently this is super-slow anyways, probably because it's reading in the entire dataset.
            in the ideal world, the dataset would be stored in separate chunks, and we could read in parallel.
        """
        if not vocab:
            vocab = MotifsExtractor.load_vocab(verbose)
        spacy_dict = {}
        iterable_docs = enumerate(MotifsExtractor.iterate_spacy(path,vocab))
        for idx, (key, doc) in iterable_docs:
            if verbose and (idx > 0) and (idx % verbose == 0):
                print('\t%03d' % idx)
            spacy_dict[key] = doc
        return spacy_dict

    @staticmethod
    def spacify(text_iter, outfile_name, spacy_NLP, verbose):
        """
            spacifies, writes a spacy object = file w/ spacy objects + other files w/ keys to said objects
            text_iter: iterates over text to spacify, yielding index and text
            outfile_name: where to write the spacy file. will write outfile_name.bin, outfile_name.txt
            if you don't want to keep loading spacy NLP objects (which takes a while) then can
                pass an existing spacy_NLP.
        """
        if verbose:
            print("Using prefix", outfile_name, "for spacy")
        if not spacy_NLP:
            if verbose:
                print('loading spacy NLP')
            spacy_NLP = spacy.load('en')
        spacy_keys = []
        spacy_objs = []
        for idx, (text_idx, text, pair_idx) in enumerate(text_iter):
            if verbose and (idx > 0) and (idx % 10000 == 0):
                print('\t%03d' % idx)
            spacy_keys.append(text_idx)
            spacy_objs.append(spacy_NLP(text))
        with open(outfile_name + '.pk', 'wb') as f:
            pickle.dump(spacy_objs, f)
        with open(outfile_name + '.txt','w') as f:
            f.write('\n'.join(spacy_keys))
 
    @staticmethod
    def deduplicate_motifs(question_fits, threshold, verbose):
        """
            Removes duplicate motifs and writes final motifs to the outfiles.
            question_fits contains the motifs to deduplicate
            outfile is the prefix for the two motif outfiles
            If two motifs co-occur in a higher proportion of cases than
            threshold, they are considered duplicates and one is removed
        """
        if verbose:
            print('\treading raw fits')
        span_to_fits = defaultdict(set)
        arcset_counts = defaultdict(int)
        for entry in question_fits:
            span_to_fits[entry['span_idx']].add(tuple(entry['arcset']))
            arcset_counts[tuple(entry['arcset'])] += 1
        if verbose:
            print('\tcounting cooccs')
        coocc_counts = defaultdict(lambda: defaultdict(int))
        for idx, (span_idx, fit_arcs) in enumerate(span_to_fits.items()):
            if verbose and (idx > 0) and (idx % verbose == 0):
                print('\t%03d' % idx)
            fit_arcs = list(fit_arcs)
            for i in range(len(fit_arcs)):
                for j in range(i+1,len(fit_arcs)):
                    arc1 = fit_arcs[i]
                    arc2 = fit_arcs[j]
                    coocc_counts[arc1][arc2] += 1
                    coocc_counts[arc2][arc1] += 1
        if verbose:
            print('\tdeduplicating')
        superset_idx = 0
        supersets = defaultdict(set)
        arcset_to_superset = {}
        for arcset, count in arcset_counts.items():
            if arcset in arcset_to_superset: continue
            arcset_to_superset[arcset] = superset_idx
            supersets[superset_idx].add(arcset)
            stack = [arc2 for arc2,count2 in coocc_counts.get(arcset,{}).items()
                        if (count2/count >= threshold) and (count2/arcset_counts[arc2] >= threshold)]
            while len(stack) > 0:
                neighbour = stack.pop()
                neighbour_count = arcset_counts[neighbour]
                arcset_to_superset[neighbour] = superset_idx
                supersets[superset_idx].add(neighbour)
                stack += [arc2 for arc2,count2 in coocc_counts.get(neighbour,{}).items()
                        if (count2/neighbour_count >= threshold) and (count2/arcset_counts[arc2] >= threshold) and (arc2 not in arcset_to_superset)]
            superset_idx += 1
        superset_ids = {}
        for idx, superset in supersets.items():
            superset_ids[idx] = sorted(superset, key=lambda x: (arcset_counts[x],len(x)), reverse=True)[0]
        arcset_to_ids = {k: superset_ids[v] for k,v in arcset_to_superset.items()}
        supersets_by_id = [{'idx': k, 'id': superset_ids[k], 'items': list(v)} for k,v in supersets.items()]
        arcset_to_super = [{'arcset': k, 'super': v} for k,v in arcset_to_ids.items()]

        return arcset_to_super, supersets_by_id

    @staticmethod
    def postprocess_fits(question_fits, tree_data, question_supersets, verbose):
        """
            Removes redundant motifs. If a pair of motifs co-occur greater than
            threshold fraction of the time (i.e. p(m1|m2), p(m2|m1) > threshold), one of them is removed.
            Writes the remaining non redundant motifs to the three files specified by the arguments.

        """
        downlinks = MotifsExtractor.read_downlinks(tree_data["downlinks"])
        super_mappings = {}
        for entry in question_supersets:
            super_mappings[tuple(entry['arcset'])] = tuple(entry['super'])
        super_counts = defaultdict(int)
        span_to_fits = defaultdict(set)
        for idx,entry in enumerate(question_fits):
            if verbose and (idx > 0) and (idx % verbose == 0):
                print('\t%03d' % idx)
            span_to_fits[entry['span_idx']].add(tuple(entry['arcset']))
        for span_idx, fit_set in span_to_fits.items():
            super_fit_set = set([super_mappings[x] for x in fit_set if x != ('*',)])
            for x in super_fit_set:
                super_counts[x] += 1
            #span_to_super_fits[span_idx] = super_fit_set
        if verbose:
            print('\tmaking new entries')
        new_entries = []
        for idx, (span_idx, fit_set) in enumerate(span_to_fits.items()):
            if verbose and (idx > 0) and (idx % verbose == 0):
                print('\t%03d' % idx)
            text_idx = QuestionTypologyUtils.get_text_idx_from_span(span_idx)
            super_to_superchildren = defaultdict(set)
            for set_ in fit_set:
                if set_ == ('*',): continue
                superset = super_mappings[set_]
                super_to_superchildren[superset].update([super_mappings[child] for child,_ in downlinks.get(set_, []) if child in fit_set])
            for superset, superchildren in super_to_superchildren.items():
                entry = {'arcset': superset, 'arcset_count': super_counts[superset],
                        'text_idx': text_idx, 'span_idx': span_idx}
                if len(superchildren) == 0:
                    entry['max_child_count'] = 0
                else:
                    entry['max_child_count'] = max(super_counts.get(child,0) for child in superchildren)
                new_entries.append(entry)

        return new_entries

    @staticmethod
    def contains_candidate(container, candidate):
        """
            True if candidate is a subset of container
        """
        return set(candidate).issubset(container)

    @staticmethod
    def fit_question(arc_set, downlinks, node_counts):
        """
            Helper to fit_all.
            Figures out the number of children of each motif in the dependency parse
        """
        fit_nodes = {}
        node_stack = [('*',)]
        i = 0
        while len(node_stack) > 0 and i < 1000:
            i += 1
            next_node = node_stack.pop()
            node_count = node_counts.get(next_node, None)
            if node_count:
                entry = {'arcset': next_node, 'arcset_count': node_count}
                children = downlinks.get(next_node, [])
                valid_children = [child for child,_ in children if MotifsExtractor.contains_candidate(arc_set, child)]

                if len(valid_children) == 0:
                    entry['max_valid_child_count'] = 0
                else:
                    entry['max_valid_child_count'] = max(node_counts.get(child,0) for child in valid_children)
                node_stack += valid_children
                fit_nodes[next_node] = entry
        return fit_nodes

    @staticmethod
    def fit_all(arc_list, tree_data, verbose):
        """
            figures out which motifs occur in each piece of text.
            arc_file: listing of arcs per text, from extract_arcs
            tree_file: the motif graph, from make_arc_tree. note that
                       this doesn't have to come from the same dataset
                       as arc_file, in which case you're basically fitting
                       a new dataset to motifs extracted elsewhere.
        """
        arc_sets = {entry["pair_idx"]: entry["arcs"] for entry in arc_list}

        downlinks = MotifsExtractor.read_downlinks(tree_data["downlinks"])
        node_counts = MotifsExtractor.read_nodecounts(tree_data["arcs"])


        if verbose:
            print('\tfitting arcsets')
        span_fit_entries = []
        for idx, (span_idx,arcs) in enumerate(arc_sets.items()):
            if verbose and (idx > 0) and (idx % verbose == 0):
                print('\t%03d' % idx)
            text_idx = QuestionTypologyUtils.get_text_idx_from_span(span_idx)
            fit_nodes = MotifsExtractor.fit_question(set(arcs), downlinks, node_counts)
            for fit_info in fit_nodes.values():
                fit_info['span_idx'] = span_idx
                fit_info['text_idx'] = text_idx
                # fit_info['pair_idx'] = pair_idx
                span_fit_entries.append(fit_info)
        return span_fit_entries

    @staticmethod
    def get_sorted_combos(itemset, k):
        """
            Returns all sorted combinations of k elements of itemset
        """
        combos = set()
        for set_ in itertools.combinations(itemset,k):
            combos.add(tuple(sorted(set_)))
        return combos

    @staticmethod
    def get_mini_powerset(itemset, k):
        """
            Returns a subset of the power set of itemset. The subset is the
            set of all subsets of itemset that are of length k or shorter
        """
        powerset = set()
        for k in range(1,min(k+1,len(itemset)+1)):
            powerset.update(MotifsExtractor.get_sorted_combos(itemset,k))
        return powerset


    @staticmethod
    def count_frequent_itemsets(arc_sets, min_support, k, verbose):
        """
            TODO
        """
        itemset_counts = defaultdict(lambda: defaultdict(int))
        span_to_itemsets = defaultdict(lambda: defaultdict(set))
        if verbose:
            print('\tfirst pass')
        for idx, (span_idx,arcs) in enumerate(arc_sets.items()):
            if verbose and (idx > 0) and (idx % verbose == 0):
                print('\t%03d' % idx)
            for itemset in MotifsExtractor.get_mini_powerset(arcs,k):
                itemset_counts[len(itemset)][itemset] += 1
                span_to_itemsets[span_idx][len(itemset)].add(itemset)

        for span_idx, count_dicts in span_to_itemsets.items():
            for i in range(1,k+1):
                count_dicts[i] = [arcset for arcset in count_dicts[i] if itemset_counts[i][arcset] >= min_support]
        if verbose:
            print('\tand then the rest')
        setsize = k+1
        spans_to_check = [span_idx for span_idx,span_dict in span_to_itemsets.items() if len(span_dict[k]) > 0]
        while len(spans_to_check) > 0:
            if verbose:
                print('\t',setsize,len(spans_to_check))
            for idx, span_idx in enumerate(spans_to_check):
                if verbose and (idx > 0) and (idx % verbose == 0):
                    print('\t%03d' % idx)
                arcs = arc_sets[span_idx]
                if len(arcs) < setsize: continue
                sets_to_check = [set_ for set_ in span_to_itemsets[span_idx].get(setsize-1,[])
                                    if itemset_counts[setsize-1].get(set_,0) >= min_support]
                if len(sets_to_check) == 0: continue

                newsets = set()
                for arc in arcs:
                    if itemset_counts[1].get((arc,),0) >= min_support:
                        for set_ in sets_to_check:
                            newset = tuple(sorted(set(set_+ (arc,))))
                            if len(newset) == setsize:
                                newsets.add(newset)
                for newset in newsets:
                    itemset_counts[setsize][newset] += 1
                    span_to_itemsets[span_idx][setsize].add(newset)
            for span_idx, count_dicts in span_to_itemsets.items():
                count_dicts[setsize] = [arcset for arcset in count_dicts[setsize] if itemset_counts[setsize][arcset] >= min_support]
            spans_to_check = [span_idx for span_idx,span_dict in span_to_itemsets.items() if len(span_dict[setsize]) > 0]
            setsize+=1
        return itemset_counts, span_to_itemsets

    @staticmethod
    def make_arc_tree(arc_list, min_support, item_set_size, verbose):
        """
            Makes the tree of motifs. (G in the paper)
        """

        arc_sets = {entry["pair_idx"]: entry["arcs"] for entry in arc_list}

        if verbose:
            print('\tcounting itemsets')
        itemset_counts, span_to_itemsets = MotifsExtractor.count_frequent_itemsets(arc_sets,min_support, item_set_size, verbose)
        new_itemset_counts = {}
        for setsize, size_dict in itemset_counts.items():
            for k,v in size_dict.items():
                if v >= min_support:
                    new_itemset_counts[k] = v
        itemset_counts = new_itemset_counts
        itemset_counts[('*',)] = len(arc_sets)
        if verbose:
            print('\twriting itemsets')
        sorted_counts = sorted(itemset_counts.items(),key=lambda x: (-x[1],len(x[0]),x[0][0]))
        arc_set_list = []
        for k,v in sorted_counts:
            arc_set_list.append((v, len(k), k))

        if verbose:
            print('\tbuilding tree')
        edges = []
        uplinks = defaultdict(dict)
        downlinks = defaultdict(dict)

        for itemset,count in itemset_counts.items():
            parents = []
            set_size = len(itemset)
            if set_size == 1:
                arc = itemset[0]
                if arc.endswith('*'):
                    parents.append(('*',))
                elif '_' in arc:
                    parents.append((arc.split('_')[0] + '_*',))
                elif '>' in arc:
                    parents.append((arc.split('>')[0] + '>*',))

            else:
                for idx in range(set_size):
                    parents.append(itemset[:idx] + itemset[idx+1:])
            for parent in parents:
                parent_count = itemset_counts[parent]
                pr_child = count / itemset_counts[parent]
                edges.append({'child': itemset, 'child_count': count,
                            'parent': parent, 'parent_count': parent_count,
                            'pr_child': pr_child})
                uplinks[itemset][parent] = {'pr_child': pr_child, 'parent_count': parent_count}
                downlinks[parent][itemset] = {'pr_child': pr_child, 'child_count': count}

        uplink_list = []
        for child, parent_dict in uplinks.items():
            uplink_list.append({'child': child, 'parents': sorted(parent_dict.items(),key=lambda x: x[1]['pr_child'])})
        uplink_list = sorted(uplink_list, key=lambda x: itemset_counts[x['child']], reverse=True)

        downlink_list = []
        for parent, child_dict in downlinks.items():
            downlink_list.append({'parent': parent, 'children': sorted(child_dict.items(),key=lambda x: x[1]['pr_child'])})
        downlink_list = sorted(downlink_list, key=lambda x: itemset_counts[x['parent']], reverse=True)

        return {"arcs": arc_set_list, "edges": edges, "uplinks": uplink_list, "downlinks": downlink_list}

    @staticmethod
    def is_noun_ish(word):
        """
            True if the word is a noun, pronoun or determiner
        """
        return (word.dep in NP_LABELS) or (word.tag_.startswith('NN') or word.tag_.startswith('PRP')) or (word.tag_.endswith('DT'))

    @staticmethod
    def has_w_det(token) -> Optional[str]:
        """
            Returns the tokens text if it has a W determiner, None otherwise
        """
        if token.tag_.startswith('W'): return token.text
        first_tok = next(token.subtree)
        if (first_tok.tag_.startswith('W')): return first_tok.text
        return None

    @staticmethod
    def get_tok(token) -> Tuple[str, bool]:
        """
            TODO
        """
        if MotifsExtractor.is_noun_ish(token):
            has_w = MotifsExtractor.has_w_det(token)
            if has_w is not None:
                return has_w.lower(), True
            else:
                return 'NN', True
        else:
            return token.text.lower(), False

    @staticmethod
    def get_clean_tok(tok):
        """
            Removes dashes from the tokens text
        """
        out_tok, is_noun = MotifsExtractor.get_tok(tok)
        return out_tok.replace('--','').strip(), is_noun

    @staticmethod
    def is_alpha_ish(text):
        """
            True if the token is comprised of only letters, or one non letter
            followed by only letters
        """
        return text.isalpha() or text[1:].isalpha()

    @staticmethod
    def is_usable(text):
        """
            True if the text is alpha_ish and is not a noun
        """
        return MotifsExtractor.is_alpha_ish(text) and (text != 'NN')

    @staticmethod
    def get_arcs(root, follow_conj):
        """
            Helper to extract_arcs.
            Returns the arcs in a given question
            follow_conj is whether conjunctions and their children should be included
            in the returned arc or not
        """
        # todo: could imagine version where nouns allowed
        arcs = set()
        root_tok, _ = MotifsExtractor.get_clean_tok(root)
        if not MotifsExtractor.is_usable(root_tok): return arcs

        arcs.add(root_tok + '_*')
        conj_elems = []
        for idx, kid in enumerate(root.children):
            if kid.dep_ in ['punct','cc']:
                continue
            elif kid.dep_ == 'conj':
                if follow_conj:
                    conj_elems.append(kid)
            else:
                kid_tok, _ = MotifsExtractor.get_clean_tok(kid)
                if MotifsExtractor.is_usable(kid_tok):
                    arcs.add(root_tok + '_' + kid_tok)

        first_elem = next(root.subtree)
        first_tok, _ = MotifsExtractor.get_clean_tok(first_elem)
        if MotifsExtractor.is_usable(first_tok):
            arcs.add(first_tok + '>*')
            try:
                second_elem = first_elem.nbor()
                second_tok, _ = MotifsExtractor.get_clean_tok(second_elem)
                if MotifsExtractor.is_usable(second_tok):
                    arcs.add(first_tok + '>' + second_tok)
            except:
                pass

        for conj_elem in conj_elems:
            arcs.update(MotifsExtractor.get_arcs(conj_elem, follow_conj))
        return arcs

    @staticmethod
    def is_utterance_question(text: str):
        """True if text is a question
        """
        return '?' in text

    @staticmethod
    def extract_arcs(text_iter, vocab, use_span, follow_conj, verbose):

        """
            extracts all arcs going out of the root in a sentence. used to find question motifs.

            text_iter: iterates over text for which arcs are extracted
            vocab: pre-loaded spacy vocabulary. if you pass None it will load vocab for you, but that's slow.
            use_span: filter to decide which sentences to use. the function takes in a spacy sentence object.
            follow_conj: whether to follow conjunctions and treat subtrees as sentences too.

        """

        arc_entries = []
        for idx, (text_idx, spacy_obj, pair_idx) in enumerate(text_iter):
            if verbose and (idx > 0) and (idx % verbose == 0):
                print('\t%03d' % idx)
            for span_idx, span in enumerate(spacy_obj.sents):
                if use_span(span.text):
                    curr_arcset = MotifsExtractor.get_arcs(span.root, follow_conj)
                    arc_entries.append({'idx': '%s%s%d' % (text_idx, span_delim, span_idx), 'arcs': list(curr_arcset),
                        'pair_idx': '%s%s%d' % (pair_idx, span_delim, span_idx)})
        return arc_entries

    @staticmethod
    def is_uppercase(x):
        """
            mainly because we otherwise get a bunch of badly parsed half-lines,
            enforce that answer sentences have to start in uppercase (reliable
            provided your data is well-formatted...)
        """
        return x.strip()[0].isupper()

    @staticmethod
    def extract_question_motifs(question_text_iter,
        question_filter_fn,
        follow_conj,
        min_question_itemset_support,
        deduplicate_threshold,
        item_set_size,
        verbose):
        """
            convenience pipeline to get question motifs. (see pipelines/extract_*_motifs for examples)
            question_text_iter: iterates over all questions
            question_filter_fn: only uses sentences in a question which corresponds to a question. can redefine.
            follow_conj: follows conjunctions to compound questions ("why...and how")
            min_question_itemset_support: the minimum number of times an itemset has to show up for the frequent itemset counter to consider it.
            deduplicate_threshold: how often two motifs co-occur (i.e. p(x|y) and p(y|x) for us to consider them redundant)
        """
        if verbose: print('running motif extraction pipeline')

        if verbose: print('loading spacy vocab')
        vocab = MotifsExtractor.load_vocab(verbose)

        if verbose: print('getting question arcs')
        q_arcs = MotifsExtractor.extract_arcs(question_text_iter, vocab, question_filter_fn, follow_conj, verbose)

        if verbose: print('making motif tree')
        tree_data = MotifsExtractor.make_arc_tree(q_arcs, min_question_itemset_support, item_set_size, verbose)

        if verbose: print('fitting motifs to questions')
        question_fits = MotifsExtractor.fit_all(q_arcs, tree_data, verbose)

        if verbose: print('handling redundant motifs')
        arcset_to_super, supersets_by_id = MotifsExtractor.deduplicate_motifs(question_fits, deduplicate_threshold, verbose)
        question_fits_super = MotifsExtractor.postprocess_fits(question_fits, tree_data, arcset_to_super, verbose)

        if verbose: print('done motif extraction')

        # to approximate the v1 behavior of having all motif info written to a single folder, we return
        # a dictionary wrapping all the motif data, with keys being generally similar to the filenames
        # used in the v1 code.
        return {
            "question_arcs": q_arcs,
            "question_fits": question_fits,
            "question_supersets_arcset_to_super": arcset_to_super,
            "question_tree_downlinks": tree_data["downlinks"],
            "question_tree_arc_set_counts": tree_data["arcs"]
        }

    @staticmethod
    def read_downlinks(downlink_data) -> Dict[Tuple, List[Tuple]]:
        """
            Returns a dicionary of parent to children nodes of the dependency parse of given input questions
        """
        downlinks = {}
        for entry in downlink_data:
            downlinks[tuple(entry['parent'])] = [(tuple(x),y) for x,y in entry['children']]
        return downlinks

    @staticmethod
    def read_nodecounts(nodecount_list: List) -> Dict[Tuple, int]:
        """
        Returns the count for each set of arcs
        """
        node_counts = {}
        for split in nodecount_list:
            count = int(split[0])
            set_size = int(split[1])
            itemset = tuple(split[2])
            node_counts[itemset] = count
        return node_counts

    @staticmethod
    def extract_answer_arcs(answer_text_iter, answer_filter_fn, follow_conj, verbose):
        """
        convenience pipeline to get answer motifs
        """

        if verbose: print('running answer arc pipeline')

        if verbose: print('loading spacy vocab')
        vocab = MotifsExtractor.load_vocab(verbose)

        if verbose: print('getting answer arcs')
        a_arcs = MotifsExtractor.extract_arcs(answer_text_iter, vocab, answer_filter_fn, follow_conj, verbose)

        if verbose: print('done answer arc extraction')
        
        return a_arcs

class QuestionClusterer:

    @staticmethod
    def read_uplinks(uplink_file):
        """
            Returns a dicionary of children to parent nodes of the dependency parse of given input questions
            uplink_file contains the parse
        """
        uplinks = {}
        with open(uplink_file) as f:
            for line in f.readlines():
                entry = json.loads(line)
                uplinks[tuple(entry['child'])] = [(tuple(x),y) for x,y in entry['parents']]
        return uplinks

    @staticmethod
    def get_motifs_per_question(question_fits, answer_arcs, supersets,
        question_threshold, answer_threshold, verbose):
        """
            Reads each of the input files and returns corresponding data structures.
            Returns question_to_fits, a dictionary question_to_leaf_fits, motif_counts, question_to_arcs, arc_counts

        """
        question_to_fits = defaultdict(set)
        question_to_leaf_fits = defaultdict(set)
        motif_counts = defaultdict(set)


        super_mappings = {}
        for entry in supersets:
            super_mappings[tuple(entry['arcset'])] = tuple(entry['super'])

        for idx, entry in enumerate(question_fits):
            if verbose and (idx > 0) and (idx % verbose == 0):
                print('\t%03d' % idx)
            motif = tuple(entry['arcset'])
            super_motif = super_mappings[motif]
            if entry['arcset_count'] < question_threshold: continue
            if entry['max_valid_child_count'] < question_threshold:
                question_to_leaf_fits[entry['text_idx']].add(super_motif)
                #if leaves_only: continue
            question_to_fits[entry['text_idx']].add(super_motif)
            motif_counts[super_motif].add(entry['text_idx'])
        motif_counts = {k:len(v) for k,v in motif_counts.items()}
        question_to_fits = {k: [x for x in v if motif_counts[x] >= question_threshold] for k,v in question_to_fits.items()}
        motif_counts = {k:v for k,v in motif_counts.items() if v >= question_threshold}
        question_to_leaf_fits = {k: [x for x in v if motif_counts.get(x,0) >= question_threshold] for k,v in question_to_leaf_fits.items()}

        question_to_arcs = defaultdict(set)
        arc_sets = {entry['pair_idx']: entry['arcs'] for entry in answer_arcs}
        arc_counts = defaultdict(int)
        for span_idx, arcs in arc_sets.items():
            question_to_arcs[QuestionTypologyUtils.get_text_idx_from_span(span_idx)].update(arcs)
            for arc in arcs:
                arc_counts[arc] += 1
        question_to_arcs = {k: [x for x in v if arc_counts[x] >= answer_threshold] for k,v in question_to_arcs.items()}
        arc_counts = {k:v for k,v in arc_counts.items() if v >= answer_threshold}

        return question_to_fits, question_to_leaf_fits, motif_counts, question_to_arcs, arc_counts

    @staticmethod
    def build_joint_matrix(question_fits, answer_arcs, supersets,
        question_threshold, answer_threshold, verbose):
        """
            Saves the matrices computed in the algorithm as numpy files
        """
        if verbose: print('\treading arcs and motifs')

        question_to_fits, question_to_leaf_fits, motif_counts, question_to_arcs, arc_counts =\
             QuestionClusterer.get_motifs_per_question(question_fits, answer_arcs,
                supersets, question_threshold, answer_threshold, verbose)
        question_term_list = list(motif_counts.keys())
        answer_term_list = list(arc_counts.keys())

        question_term_to_idx = {k:idx for idx,k in enumerate(question_term_list)}
        answer_term_to_idx = {k:idx for idx,k in enumerate(answer_term_list)}

        if verbose: print('\tbuilding matrices')
        question_term_idxes = []
        question_leaves = []
        question_doc_idxes = []
        answer_term_idxes = []
        answer_doc_idxes = []
        pair_idx_list = []

        pair_idxes = list(set(question_to_fits.keys()).intersection(set(question_to_arcs.keys())))

        for idx, p_idx in enumerate(pair_idxes):
            if verbose and (idx > 0) and (idx % verbose == 0):
                print('\t%03d' % idx)

            question_terms = question_to_fits[p_idx]
            answer_terms = question_to_arcs[p_idx]
            pair_idx_list.append(p_idx)

            for term in question_terms:
                term_idx = question_term_to_idx[term]
                question_term_idxes.append(term_idx)
                question_doc_idxes.append(idx)
                question_leaves.append(term in question_to_leaf_fits.get(p_idx,[]))
            for term in answer_terms:
                term_idx = answer_term_to_idx[term]
                answer_term_idxes.append(term_idx)
                answer_doc_idxes.append(idx)

        mtx_obj = dict()
        mtx_obj["q_tidxes"] = np.asarray(question_term_idxes)
        mtx_obj["q_leaves"] = np.asarray(question_leaves)
        mtx_obj["a_tidxes"] = np.asarray(answer_term_idxes)
        mtx_obj["q_didxes"] = np.asarray(question_doc_idxes)
        mtx_obj["a_didxes"] = np.asarray(answer_doc_idxes)

        mtx_obj['q_terms'] = []
        mtx_obj['q_term_to_idx'] = dict()
        mtx_obj['q_term_counts'] = []
        for idx, term in enumerate(question_term_list):
            mtx_obj['q_term_counts'].append(motif_counts[term])
            mtx_obj['q_terms'].append(term)
            mtx_obj['q_term_to_idx'][term] = idx
        mtx_obj['q_terms'] = np.array(mtx_obj['q_terms'])
        mtx_obj['q_term_counts'] = np.array(mtx_obj['q_term_counts'])

        mtx_obj['a_terms'] = []
        mtx_obj['a_term_to_idx'] = {}
        mtx_obj['a_term_counts'] = []
        for idx, term in enumerate(answer_term_list):
            mtx_obj['a_term_counts'].append(arc_counts[term])
            mtx_obj['a_terms'].append(term)
            mtx_obj['a_term_to_idx'][term] = idx
        mtx_obj['a_terms'] = np.array(mtx_obj['a_terms'])
        mtx_obj['a_term_counts'] = np.array(mtx_obj['a_term_counts'])

        mtx_obj['docs'] = []
        mtx_obj['doc_to_idx'] = {}
        for idx, doc_id in enumerate(pair_idxes):
            mtx_obj['docs'].append(doc_id)
            mtx_obj['doc_to_idx'][doc_id] = idx
        mtx_obj['docs'] = np.array(mtx_obj['docs'])

        return mtx_obj

    @staticmethod
    def load_joint_mtx(rootname, verbose):
        """
            Reads the saved matrix files and returns a data structure that contains
            all the information in them
            mtx_obj has the following keys:
            "q_tidxes": a list of the question term indices
            "q_leaves": a list of bools for each term. True means the term is a sink motif.
            "a_tidxes": a list of the answer term indices
            "q_didxes": a list of the question doc indices
            "a_didxes": a list of the answer doc indices
            "q_terms": a list of tuples containing the actual extracted motifs
            "q_term_to_idx": TODO
            "q_term_counts": a list of the number of occurences of each motif
            "a_terms": a list of tuples containing the actual extracted answer fragments
            "a_term_to_idx": TODO
            "a_term_counts": a list of the number of occurences of each answer fragment
            "docs": a list of length num_questions. Each entry contains the spacy document that a
                given question is stored in
            "doc_to_idx": TODO

        """
        mtx_obj = {}

        if verbose: print('reading question tidxes')
        mtx_obj['q_tidxes'] = np.load(rootname + '.q.tidx.npy')
        if verbose: print('reading question leaves')
        mtx_obj['q_leaves'] = np.load(rootname + '.q.leaves.npy')
        if verbose: print('reading answer tidxes')
        mtx_obj['a_tidxes'] = np.load(rootname + '.a.tidx.npy')

        if verbose: print('reading question didxes')
        mtx_obj['q_didxes'] = np.load(rootname + '.q.didx.npy')
        if verbose: print('reading answer didxes')
        mtx_obj['a_didxes'] = np.load(rootname + '.a.didx.npy')

        if verbose: print('reading question terms')
        mtx_obj['q_terms'] = []
        mtx_obj['q_term_to_idx'] = {}
        mtx_obj['q_term_counts'] = []
        fname = rootname + '.q.terms.txt'
        with open(fname) as f:
            for idx, line in enumerate(f.readlines()):
                count,term = line.split('\t')
                term = term.strip()
                term = make_tuple(term)
                mtx_obj['q_term_counts'].append(int(count))
                mtx_obj['q_terms'].append(term)
                mtx_obj['q_term_to_idx'][term] = idx
        mtx_obj['q_terms'] = np.array(mtx_obj['q_terms'])
        mtx_obj['q_term_counts'] = np.array(mtx_obj['q_term_counts'])

        if verbose: print('reading answer terms')
        mtx_obj['a_terms'] = []
        mtx_obj['a_term_to_idx'] = {}
        mtx_obj['a_term_counts'] = []
        fname = rootname + '.a.terms.txt'
        with open(fname) as f:
            for idx, line in enumerate(f.readlines()):
                count,term = line.split('\t')
                term = term.strip()
                mtx_obj['a_term_counts'].append(int(count))
                mtx_obj['a_terms'].append(term)
                mtx_obj['a_term_to_idx'][term] = idx
        mtx_obj['a_terms'] = np.array(mtx_obj['a_terms'])
        mtx_obj['a_term_counts'] = np.array(mtx_obj['a_term_counts'])

        if verbose: print('reading docs')
        mtx_obj['docs'] = []
        mtx_obj['doc_to_idx'] = {}
        with open(rootname + '.docs.txt') as f:
            for idx, line in enumerate(f.readlines()):
                doc_id = line.strip()
                mtx_obj['docs'].append(doc_id)
                mtx_obj['doc_to_idx'][doc_id] = idx
        mtx_obj['docs'] = np.array(mtx_obj['docs'])

        if verbose: print('done!')
        return mtx_obj

    @staticmethod
    def build_mtx(mtx_obj, data_type, norm, idf, leaves_only):
        """
            Returns mtx which is a num_motifs X num_questions matrix that represents the input
            questions/answers as per the algorithm explained in section 5 of the paper
            data_type determines whether it returns the question matrix or answer matrix
        """
        #norm = l2, idf = False, leaves_only = True
        N_terms = len(mtx_obj[data_type + '_terms'])
        N_docs = len(mtx_obj['docs'])
        if idf:
            data = np.log(N_docs) - np.log(mtx_obj[data_type + '_term_counts'][mtx_obj[data_type + '_tidxes']])
        else:
            data = np.ones_like(mtx_obj[data_type + '_tidxes'])
            if leaves_only:
                data[~mtx_obj[data_type + '_leaves']] = 0
        mtx = sparse.csr_matrix((data, (mtx_obj[data_type + '_tidxes'], mtx_obj[data_type + '_didxes'])),
            shape=(N_terms,N_docs))
        if norm:
            mtx = Normalizer(norm=norm).fit_transform(mtx.astype(np.double))

        return mtx

    @staticmethod
    def run_simple_pipe(mtx_obj, verbose: bool, norm, idf, leaves_only):
        """
            Create and return q_mtx and a_mtx from precomputed mtx_obj.
            mtx_obj has the following keys
            q_mtx and a_mtx are the question and answer matrix from the paper
        """
        q_mtx = QuestionClusterer.build_mtx(mtx_obj, 'q', norm, idf, leaves_only)
        a_mtx = QuestionClusterer.build_mtx(mtx_obj, 'a', norm, True, leaves_only)
        return q_mtx, a_mtx

    @staticmethod
    def do_sparse_svd(mtx, k):
        """
            Computes the largest k singular values/vectors for a mtx with shape M X N.
            returns u, a M X k unitary matrix having left singular vectors as columns,
            s a K X 1 vector of the singular values.
            and v a k X N unitary matrix having right singular vectors as rows

        """
        u,s,v = sparse.linalg.svds(mtx, k=k)
        return u[:,::-1],s[::-1],v[::-1,:]

    @staticmethod
    def run_lowdim_pipe(q_mtx, a_mtx, k, snip):
        """
            projects q_mtx and a_mtx to the latent space as described in the paper.
            k is the number of singular values for the SVD decomposition.
            snip is True if the results should be returned with the first dimension removed
        """
        a_u, a_s, a_v = QuestionClusterer.do_sparse_svd(a_mtx,k + int(snip))
        lq = q_mtx * (a_v.T * a_s**-1)
        if snip:
            return QuestionClusterer.snip_first_dim(lq, a_u, a_s, a_v)
        else:
            return lq, a_u, a_s, a_v

    @staticmethod
    def inspect_latent_space(mtx, names, dim_iter=None, num_dims=5, num_egs=10, which_end=None, skip_first=True, dim_names=None,s=None):
        if dim_names is None: dim_names = {}
        mtx = Normalizer().fit_transform(mtx).T
        if dim_iter is None:
            dim_iter = range(int(skip_first), num_dims + int(skip_first))
        for dim in dim_iter:
            if s is not None:
                print(dim,s[dim])
            else:
                print(dim)
            row = mtx[dim]
            argsorted = np.argsort(row)
            if (not which_end) or (which_end == -1):
                print('\tbottom',dim_names.get((dim,-1), ''))
                for i in range(num_egs):
                    print('\t\t',names[argsorted[i]], '%+.4f' % row[argsorted[i]])
            if (not which_end) or (which_end == 1):
                print('\ttop',dim_names.get((dim,1), ''))
                for i in range(num_egs):
                    print('\t\t',names[argsorted[-1-i]], '%+.4f' % row[argsorted[-1-i]])
            print()

    @staticmethod
    def run_kmeans(X, k, max_iter, random_seed):
        """
            runs a Kmeans clustering algorithm with X as inputs.
            k is the number of clusters
            max_iter is the number of iterations the clustering algorithm should run for\
            random_seed ensures that the same clusters are produced if the same random seed is supplied again.

        """
        km = KMeans(n_clusters=k, max_iter=max_iter, n_init=1000, random_state=random_seed)
        km.fit(X)
        return km

    @staticmethod
    def inspect_kmeans_run(q_mtx, a_mtx, num_svd_dims, num_clusters, q_terms,
        a_terms, km, remove_first, max_iter, random_seed):
        """
            Runs the clustering algorithm and returns a sklearn.cluster.KMeans object that can be used
            to classify new inputs and a dictionary types_to_data with the following keys:
            its keys are the indices of the clusters (here 0-7).
            The values are dictionaries with the following keys:

            "motifs": the motifs, as a list of tuples of the motif terms
            "motif_dists": the corresponding distances of each motif from the centroid of the cluster this
                motif is in
            "fragments": the answer fragments, as a list of tuples of answer terms
            "fragment_dists": the corresponding distances of each fragment from the centroid of the
                cluster this fragment is in
            "questions": the IDs of the questions in this cluster. You can get the corresponding
                question text by using the get_question_text_from_pair_idx(pair_idx) method.
            "question_dists": the corresponding distances of each question from the centroid of the cluster
                this question is in
        """
        if remove_first:
            q_mtx = q_mtx[:,1:(num_svd_dims + 1)]
            a_mtx = a_mtx[:,1:(num_svd_dims + 1)]
        else:
            q_mtx = q_mtx[:,:num_svd_dims]
            a_mtx = a_mtx[:,:num_svd_dims]
        q_mtx = Normalizer().fit_transform(q_mtx)
        a_mtx = Normalizer().fit_transform(a_mtx)
        types_to_data = {}
        if km:
            q_km = km
        else:
            q_km = QuestionClusterer.run_kmeans(q_mtx, num_clusters, max_iter, random_seed)

        q_dists = q_km.transform(q_mtx)
        q_assigns = q_km.labels_
        a_dists = q_km.transform(a_mtx)
        a_assigns = q_km.predict(a_mtx)
        for cl in range(num_clusters):
            types_to_data[cl] = {
                "motifs": [],
                "motif_dists": [],
                "fragments": [],
                "fragment_dists": [],
                "questions": [],
                "question_dists": [],
            }
            q_assigned = q_assigns == cl
            median_qdist = np.median(q_dists[:,cl][q_assigned])
            a_assigned = a_assigns == cl
            median_adist = np.median(a_dists[:,cl][a_assigned])
            argsorted_qdists = np.argsort(q_dists[:,cl])
            argsorted_qdists = argsorted_qdists[np.in1d(argsorted_qdists, np.where(q_assigned)[0])]
            for i in range(q_assigned.sum()):
                curr_qdist = q_dists[:,cl][argsorted_qdists[i]]
                if curr_qdist > median_qdist:
                    diststr = '%.4f ~~' %  curr_qdist
                else:
                    diststr = '%.4f' % curr_qdist
                types_to_data[cl]["motifs"].append(q_terms[argsorted_qdists[i]])
                types_to_data[cl]["motif_dists"].append(diststr)
            argsorted_adists = np.argsort(a_dists[:,cl])
            argsorted_adists = argsorted_adists[np.in1d(argsorted_adists, np.where(a_assigned)[0])]
            for i in range(a_assigned.sum()):
                curr_adist = a_dists[:,cl][argsorted_adists[i]]
                if curr_adist > median_adist:
                    diststr = '%.4f ~~' %  curr_adist
                else:
                    diststr = '%.4f' % curr_adist
                types_to_data[cl]["fragments"].append(a_terms[argsorted_adists[i]])
                types_to_data[cl]["fragment_dists"].append(diststr)
        return q_km, types_to_data

    @staticmethod
    def snip_first_dim(lq, a_u, a_s, a_v):
        """
            Returns each of the input matrices with the first column snipped off
        """
        return lq[:,1:], a_u[:,1:], a_s[1:], a_v[1:]

    @staticmethod
    def assign_clusters(km, lq, a_u, mtx_obj, n_dims, norm, idf, leaves_only):
        """
            Assigns correct type to each of the questions in the training data
            Returns motif_df, aarc_df, qdoc_df, q_leaves and qdoc_vects
            motif_df: a dictionary that has information about the motifs and which clusters they were assigned
            aarc_df: a dictionary that has information about the answer fragments and which clusters they were assigned
            qdoc_df: a dictionary that has information about the questions and which clusters they were assigned
            q_leaves: TODO
            qdoc_vects: the vectors denoting each question in the latent space
        """
        km_qdists = km.transform(Normalizer().fit_transform(lq[:,:n_dims]))
        km_qlabels = km.predict(Normalizer().fit_transform(lq[:,:n_dims]))
        km_adists = km.transform(Normalizer().fit_transform(a_u[:,:n_dims]))
        km_alabels = km.predict(Normalizer().fit_transform(a_u[:,:n_dims]))

        motif_df_entries = []
        for idx, motif in enumerate(mtx_obj['q_terms']):
            entry = {'idx': idx, 'motif': motif, 'cluster': km_qlabels[idx],
                    'count': mtx_obj['q_term_counts'][idx]}
            entry['cluster_dist'] = km_qdists[idx,entry['cluster']]
            motif_df_entries.append(entry)
        motif_df = pd.DataFrame(motif_df_entries).set_index('idx')

        aarc_df_entries = []
        for idx, aarc in enumerate(mtx_obj['a_terms']):
            entry = {'idx': idx, 'aarc': aarc, 'cluster': km_alabels[idx],
                    'count': mtx_obj['a_term_counts'][idx]}
            entry['cluster_dist'] = km_adists[idx,entry['cluster']]
            aarc_df_entries.append(entry)
        aarc_df = pd.DataFrame(aarc_df_entries).set_index('idx')

        q_leaves = QuestionClusterer.build_mtx(mtx_obj,'q', norm, idf, leaves_only)
        qdoc_vects = Normalizer().fit_transform(q_leaves.T) * Normalizer().fit_transform(lq)
        km_qdoc_dists = km.transform(Normalizer().fit_transform(qdoc_vects[:,:n_dims]))
        km_qdoc_labels = km.predict(Normalizer().fit_transform(qdoc_vects[:,:n_dims]))
        qdoc_df_entries = []
        for idx, qdoc in enumerate(mtx_obj['docs']):
            entry = {'idx': idx, 'q_idx': qdoc, 'cluster': km_qdoc_labels[idx]}
            entry['cluster_dist'] = km_qdoc_dists[idx,entry['cluster']]
            entry['all_cluster_dists'] = km_qdoc_dists[idx,:]
            qdoc_df_entries.append(entry)
        qdoc_df = pd.DataFrame(qdoc_df_entries).set_index('idx')

        return motif_df, aarc_df, qdoc_df, q_leaves, qdoc_vects

    @staticmethod
    def build_matrix(motifs, question_threshold, answer_threshold, verbose):
        """
            convenience pipeline to build the question answer matrices.
            motif_dir: wherever extract_motifs wrote to
            question_threshold: minimum # of questions in which a question motif has to occur to be considered
            answer_threshold: minimum # of answers in which a fragment has to occur to be considered
        """
        if verbose: print('building q-a matrices')
        question_fits = motifs['question_fits']
        answer_arcs = motifs['answer_arcs']
        supersets = motifs['question_supersets_arcset_to_super']

        return QuestionClusterer.build_joint_matrix(question_fits, answer_arcs, supersets,
            question_threshold, answer_threshold, verbose)

    @staticmethod
    def extract_clusters(mtx_obj, k, d, snip, verbose, norm, idf, leaves_only,
        remove_first, max_iter, random_seed):
        """
            convenience pipeline to get latent q-a dimensions and clusters.

            km_file: where to write the kmeans object
            k: num clusters
            d: num latent dims

        """
        q_mtx, a_mtx = QuestionClusterer.run_simple_pipe(mtx_obj, verbose,
            norm, idf, leaves_only)
        lq, a_u, a_s, a_v = QuestionClusterer.run_lowdim_pipe(q_mtx, a_mtx,d, snip)
        km, types_to_data = QuestionClusterer.inspect_kmeans_run(lq, a_u, d, k, mtx_obj['q_terms'],
            mtx_obj['a_terms'], None, remove_first, max_iter, random_seed)

        return km, types_to_data, lq, a_u, a_s, a_v

class QuestionTypologyUtils:

    @staticmethod
    def read_arcs(arc_file, verbose):
        """
            Reads the files containing question arcs from the input questions and returns arc_sets,
            a dicionary whose keys are the question indexes and values are the arcs in these questions
        """
        arc_sets = {}
        with open(arc_file) as f:
            for idx,line in enumerate(f.readlines()):
                if verbose and (idx > 0) and (idx % verbose == 0):
                    print('\t%03d' % idx)
                entry = json.loads(line)
                arc_sets[entry['pair_idx']] = entry['arcs']
        return arc_sets

    @staticmethod
    def get_text_idx_from_span(span_idx):
        """
            Given the index of a span within a question, return the index of the entire question
        """
        return span_idx[:span_idx.rfind(span_delim)]

    @staticmethod
    def get_q_idx_from_pair(pair_idx):
        """
            Given the index of a question-answer pair, return the index of the question
        """
        return pair_idx[:pair_idx.rfind(pair_delim)]
