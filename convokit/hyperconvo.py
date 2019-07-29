"""Implements the hypergraph conversation model from
http://www.cs.cornell.edu/~cristian/Patterns_of_participant_interactions.html.

Example usage: hypergraph creation and feature extraction, visualization and interpretation on a subsample of Reddit.
(https://github.com/CornellNLP/Cornell-Conversational-Analysis-Toolkit/blob/master/examples/hyperconvo/demo.ipynb)
"""

import itertools
from collections import defaultdict
import numpy as np
import scipy.stats
from .transformer import Transformer
from typing import Tuple, List, Dict, Optional, Hashable, Collection
from .model import Corpus, Utterance
from .triadMotif import TriadMotif, MotifType

class Hypergraph:
    """
    Represents a hypergraph, consisting of nodes, directed edges,
    hypernodes (each of which is a set of nodes) and hyperedges (directed edges
    from hypernodes to hypernodes). Contains functionality to extract motifs
    from hypergraphs (Fig 2 of
    http://www.cs.cornell.edu/~cristian/Patterns_of_participant_interactions.html)
    """
    def __init__(self):
        # public
        self.nodes = dict()
        self.hypernodes = dict()

        # private
        self.adj_out = dict()  # out edges for each (hyper)node
        self.adj_in = dict()   # in edges for each (hyper)node

    def add_node(self, u: Hashable, info: Optional[Dict]=None) -> None:
        self.nodes[u] = info if info is not None else dict()
        self.adj_out[u] = dict()
        self.adj_in[u] = dict()

    def add_hypernode(self, name: Hashable,
                      nodes: Collection[Hashable],
                      info: Optional[dict]=None) -> None:
        self.hypernodes[name] = set(nodes)
        self.adj_out[name] = dict()
        self.adj_in[name] = dict()

    # edge or hyperedge
    def add_edge(self, u: Hashable, v: Hashable, info: Optional[dict]=None) -> None:
        assert u in self.nodes or u in self.hypernodes
        assert v in self.nodes or v in self.hypernodes
        if u in self.hypernodes and v in self.hypernodes:
            assert len(info.keys()) > 0
        if v not in self.adj_out[u]:
            self.adj_out[u][v] = []
        if u not in self.adj_in[v]:
            self.adj_in[v][u] = []
        if info is None: info = dict()
        self.adj_out[u][v].append(info)
        self.adj_in[v][u].append(info)

    def edges(self) -> Dict[Tuple[Hashable, Hashable], List]:
        return dict(((u, v), lst) for u, d in self.adj_out.items()
                           for v, lst in d.items())

    def outgoing_nodes(self, u: Hashable) -> Dict[Hashable, List]:
        assert u in self.adj_out
        return dict((v, lst) for v, lst in self.adj_out[u].items()
                           if v in self.nodes)

    def outgoing_hypernodes(self, u) -> Dict[Hashable, List]:
        assert u in self.adj_out
        return dict((v, lst) for v, lst in self.adj_out[u].items()
                           if v in self.hypernodes)

    def incoming_nodes(self, v: Hashable) -> Dict[Hashable, List]:
        assert v in self.adj_in
        return dict((u, lst) for u, lst in self.adj_in[v].items() if u in
                           self.nodes)

    def incoming_hypernodes(self, v: Hashable) -> Dict[Hashable, List]:
        assert v in self.adj_in
        return dict((u, lst) for u, lst in self.adj_in[v].items() if u in
                           self.hypernodes)

    def outdegrees(self, from_hyper: bool=False, to_hyper: bool=False) -> List[int]:
        return [sum([len(l) for v, l in self.adj_out[u].items() if v in
                     (self.hypernodes if to_hyper else self.nodes)]) for u in
                (self.hypernodes if from_hyper else self.nodes)]

    def indegrees(self, from_hyper: bool=False, to_hyper: bool=False) -> List[int]:
        return [sum([len(l) for u, l in self.adj_in[v].items() if u in
                     (self.hypernodes if from_hyper else self.nodes)]) for v in
                (self.hypernodes if to_hyper else self.nodes)]

    @staticmethod
    def _sorted_ts(timestamps: List[Dict[str, int]]) -> List[Dict[str, int]]:
        """
        helper method for getting sorted timestamps of edges between hypernodes, e.g. from Hypergraph.adj_out[C1][C2]
        :param timestamps: e.g. [{'timestamp': 1322706222, 'text': "Lolapalooza"}, {'timestamp': 1322665765, 'text': "Wanda"}]
        :return: edge dictionaries sorted by timestamp
        """
        return sorted(timestamps, key=lambda x: x['timestamp'])

    def extract_motifs(self) -> Dict[str, List]:

        motifs = dict()

        for motif_type, motif_func in [
            (MotifType.NO_EDGE_TRIADS.name, self.no_edge_triad_motifs),
            (MotifType.SINGLE_EDGE_TRIADS.name, self.single_edge_triad_motifs),
            (MotifType.INCOMING_TRIADS.name, self.incoming_triad_motifs),
            (MotifType.OUTGOING_TRIADS.name, self.outgoing_triad_motifs),
            (MotifType.DYADIC_TRIADS.name, self.dyadic_triad_motifs),
            (MotifType.UNIDIRECTIONAL_TRIADS.name, self.unidirectional_triad_motifs),
            (MotifType.INCOMING_2TO3_TRIADS.name, self.incoming_2to3_triad_motifs),
            (MotifType.INCOMING_1TO3_TRIADS.name, self.incoming_1to3_triad_motifs),
            (MotifType.DIRECTED_CYCLE_TRIADS.name, self.directed_cycle_triad_motifs),
            (MotifType.OUTGOING_3TO1_TRIADS.name, self.outgoing_3to1_triad_motifs),
            (MotifType.INCOMING_RECIPROCAL_TRIADS.name, self.incoming_reciprocal_triad_motifs),
            (MotifType.OUTGOING_RECIPROCAL_TRIADS.name, self.outgoing_reciprocal_motifs),
            (MotifType.DIRECTED_CYCLE_1TO3_TRIADS.name, self.directed_cycle_1to3_triad_motifs),
            (MotifType.DIRECIPROCAL_TRIADS.name, self.direciprocal_triad_motifs),
            (MotifType.DIRECIPROCAL_2TO3_TRIADS.name, self.direciprocal_2to3_triad_motifs),
            (MotifType.TRIRECIPROCAL_TRIADS.name, self.trireciprocal_triad_motifs)
        ]:
            motifs[motif_type] = motif_func()

        return motifs

    # returns list of tuples of form (C1, C2, C3), no edges
    def no_edge_triad_motifs(self):
        motifs = []
        for C1, C2, C3 in itertools.combinations(self.hypernodes, 3):
            if C1 not in self.adj_in[C2] and C1 not in self.adj_in[C3]:
                if C2 not in self.adj_in[C3] and C2 not in self.adj_in[C1]:
                    if C3 not in self.adj_in[C1] and C3 not in self.adj_in[C2]:
                        motifs += [TriadMotif((C1, C2, C3), (), MotifType.NO_EDGE_TRIADS.name)]
        return motifs

    # returns list of tuples of form (C1, C2, C3, C1->C2)
    def single_edge_triad_motifs(self):
        motifs = []
        for C1 in self.hypernodes:
            outgoing = set(self.outgoing_hypernodes(C1))
            incoming = set(self.incoming_hypernodes(C1))
            non_adjacent = set(self.hypernodes) - outgoing.union(incoming)
            outgoing_only = outgoing - incoming

            motifs += [TriadMotif((C1, C2, C3), (Hypergraph._sorted_ts(self.adj_out[C1][C2]),), MotifType.SINGLE_EDGE_TRIADS.name)
                       for C2 in outgoing_only
                       for C3 in non_adjacent if ((C3 not in self.adj_out[C2]) and (C3 not in self.adj_in[C2]))]
        return motifs

    # returns list of tuples of form (C1, C2, C3, C1->C2, C2->C1)
    def dyadic_triad_motifs(self):
        motifs = []
        for C3 in self.hypernodes: # define the triad with respect to C3 <- prevents double counting
            outgoing = set(self.outgoing_hypernodes(C3))
            incoming = set(self.incoming_hypernodes(C3))
            non_adjacent = set(self.hypernodes) - outgoing.union(incoming)

            motifs += [TriadMotif((C1, C2, C3),
                                  (Hypergraph._sorted_ts(self.adj_out[C1][C2]),
                                   Hypergraph._sorted_ts(self.adj_out[C2][C1])),
                                  MotifType.DYADIC_TRIADS.name)
                       for C1, C2 in itertools.combinations(non_adjacent, 2)
                       if ((C2 in self.adj_out[C1]) and (C1 in self.adj_out[C2]))]
        return motifs


    # returns list of tuples of form (C1, C2, C1->C2, C2->C1) as in paper
    def dyadic_interaction_motifs(self):
        motifs = []
        for C1 in self.hypernodes:
            motifs += [(C1, C2, e1, e2) for C2 in self.adj_out[C1] if C2 in
                       self.hypernodes and C1 in self.adj_out[C2]
                       for e1 in self.adj_out[C1][C2]
                       for e2 in self.adj_out[C2][C1]]
        return motifs

    # returns list of tuples of form (C1, C2, C3, C2->C1, C3->C1)
    def incoming_triad_motifs(self):
        motifs = []
        for C1 in self.hypernodes:
            incoming = set(self.incoming_hypernodes(C1))
            outgoing = set(self.outgoing_hypernodes(C1))
            incoming_only = incoming - outgoing
            motifs += [TriadMotif((C1, C2, C3),
                                  (Hypergraph._sorted_ts(self.adj_out[C2][C1]), Hypergraph._sorted_ts(self.adj_out[C3][C1])),
                                  MotifType.INCOMING_TRIADS.name)
                       for C2, C3 in itertools.combinations(incoming_only, 2)]
        return motifs

    # returns list of tuples of form (C1, C2, C3, C1->C2, C1->C3)
    def outgoing_triad_motifs(self):
        motifs = []
        for C1 in self.hypernodes:
            outgoing = set(self.outgoing_hypernodes(C1))
            incoming = set(self.incoming_hypernodes(C1))
            outgoing_only = outgoing - incoming
            motifs += [TriadMotif((C1, C2, C3),
                                  (Hypergraph._sorted_ts(self.adj_out[C1][C2]),
                                   Hypergraph._sorted_ts(self.adj_out[C1][C3])),
                                  MotifType.OUTGOING_TRIADS.name)
                       for C2, C3 in itertools.combinations(outgoing_only, 2)]
        return motifs

    # returns list of tuples of form (C1, C2, C3, C1->C2, C2->C3)
    def unidirectional_triad_motifs(self):
        motifs = []
        for C2 in self.hypernodes: # define the motif with respect to C2
            incoming = set(self.incoming_hypernodes(C2))
            outgoing = set(self.outgoing_hypernodes(C2))
            incoming_only = incoming - outgoing # ignore edges C2->C1
            outgoing_only = outgoing - incoming # ignore edges C3->C2
            for C1 in incoming_only:
                for C3 in outgoing_only:
                    # ensure C3 and C1 have no edges between them
                    if C1 in self.adj_out[C3]: continue
                    if C3 in self.adj_out[C1]: continue
                    motifs += [TriadMotif((C1, C2, C3),
                                          (Hypergraph._sorted_ts(self.adj_out[C1][C2]),
                                           Hypergraph._sorted_ts(self.adj_out[C2][C3])),
                                          MotifType.UNIDIRECTIONAL_TRIADS.name)]

        return motifs

    # returns list of tuples of form (C1, C2, C3, C2->C1, C3->C1, C2->C3)
    def incoming_2to3_triad_motifs(self):
        motifs = []
        for C1 in self.hypernodes:
            incoming = set(self.incoming_hypernodes(C1))
            outgoing = set(self.outgoing_hypernodes(C1))
            incoming_only = incoming - outgoing # no edges C2->C1
            for C2, C3 in itertools.permutations(incoming_only, 2): # permutations because non-symmetric
                if C2 in self.adj_out[C3]: continue # ensure no C3->C2
                if C3 not in self.adj_out[C2]: continue # ensure C2->C3 exists
                motifs += [TriadMotif((C1, C2, C3),
                                      (Hypergraph._sorted_ts(self.adj_out[C2][C1]),
                                       Hypergraph._sorted_ts(self.adj_out[C3][C1]),
                                       Hypergraph._sorted_ts(self.adj_out[C2][C3])),
                                      MotifType.INCOMING_2TO3_TRIADS.name)
                           ]
        return motifs

    # returns list of tuples of form (C1, C2, C3, C1->C2, C2->C3, C3->C1)
    def directed_cycle_triad_motifs(self):
        # not efficient
        motifs = []
        for C1, C2, C3 in itertools.combinations(self.hypernodes, 3):
            if C3 in self.adj_out[C1]: continue
            if C1 in self.adj_out[C2]: continue
            if C2 in self.adj_out[C3]: continue

            if C2 not in self.adj_out[C1]: continue
            if C3 not in self.adj_out[C2]: continue
            if C1 not in self.adj_out[C3]: continue
            motifs += [TriadMotif((C1, C2, C3),
                                  (Hypergraph._sorted_ts(self.adj_out[C1][C2]),
                                   Hypergraph._sorted_ts(self.adj_out[C2][C3]),
                                   Hypergraph._sorted_ts(self.adj_out[C3][C1])),
                                  MotifType.DIRECTED_CYCLE_TRIADS.name)]

        return motifs

    # returns list of tuples of form (C1, C2, C3, C2->C1, C3->C1, C1->C3)
    def incoming_1to3_triad_motifs(self):
        motifs = []
        for C1 in self.hypernodes:
            incoming = set(self.incoming_hypernodes(C1))
            outgoing = set(self.outgoing_hypernodes(C1))
            incoming_only = incoming - outgoing # no edges C2->C1
            for C2, C3 in itertools.permutations(incoming_only, 2):
                if C2 in self.adj_out[C1]: continue
                if C2 in self.adj_out[C3]: continue
                if C3 in self.adj_out[C2]: continue

                if C3 not in self.adj_out[C1]: continue

                motifs += [TriadMotif((C1, C2, C3),
                                      (Hypergraph._sorted_ts(self.adj_out[C2][C1]),
                                       Hypergraph._sorted_ts(self.adj_out[C3][C1]),
                                       Hypergraph._sorted_ts(self.adj_out[C1][C3])),
                                      MotifType.INCOMING_1TO3_TRIADS.name)
                           ]

        return motifs

    # returns list of tuples of form (C1, C2, C3, C1->C2, C1->C3, C3->C1)
    def outgoing_3to1_triad_motifs(self):
        motifs = []
        for C1 in self.hypernodes:
            outgoing = self.outgoing_hypernodes(C1)
            for C2, C3 in itertools.permutations(outgoing, 2):
                if C1 in self.adj_out[C2]: continue
                if C2 in self.adj_out[C3]: continue
                if C3 in self.adj_out[C2]: continue

                if C1 not in self.adj_out[C3]: continue
                motifs += [TriadMotif((C1, C2, C3),
                                      (Hypergraph._sorted_ts(self.adj_out[C1][C2]),
                                       Hypergraph._sorted_ts(self.adj_out[C1][C3]),
                                       Hypergraph._sorted_ts(self.adj_out[C3][C1])),
                                      MotifType.OUTGOING_3TO1_TRIADS.name)
                           ]

        return motifs

    # returns list of tuples of form (C1, C2, C3, C2->C1, C3->C1, C2->C3, C3->C2)
    def incoming_reciprocal_triad_motifs(self):
        motifs = []
        for C1 in self.hypernodes:
            incoming = set(self.incoming_hypernodes(C1))
            outgoing = set(self.outgoing_hypernodes(C1))
            incoming_only = incoming - outgoing

            motifs += [TriadMotif((C1, C2, C3),
                                  (Hypergraph._sorted_ts(self.adj_out[C2][C1]),
                                   Hypergraph._sorted_ts(self.adj_out[C3][C1]),
                                   Hypergraph._sorted_ts(self.adj_out[C2][C3]),
                                   Hypergraph._sorted_ts(self.adj_out[C3][C2])),
                                  MotifType.INCOMING_RECIPROCAL_TRIADS.name)
                       for C2, C3 in itertools.combinations(incoming_only, 2)
                       if ((C3 in self.adj_out[C2]) and (C2 in self.adj_out[C3]))
                       ]
        return motifs

    # returns list of tuples of form (C1, C2, C3, C1->C2, C1->C3, C2->C3, C3->C2)
    def outgoing_reciprocal_motifs(self):
        motifs = []
        for C1 in self.hypernodes:
            incoming = set(self.incoming_hypernodes(C1))
            outgoing = set(self.outgoing_hypernodes(C1))
            outgoing_only = outgoing - incoming

            motifs += [TriadMotif((C1, C2, C3),
                                  (Hypergraph._sorted_ts(self.adj_out[C1][C2]),
                                   Hypergraph._sorted_ts(self.adj_out[C1][C3]),
                                   Hypergraph._sorted_ts(self.adj_out[C2][C3]),
                                   Hypergraph._sorted_ts(self.adj_out[C3][C2])),
                                  MotifType.OUTGOING_RECIPROCAL_TRIADS.name)
                       for C2, C3 in itertools.combinations(outgoing_only, 2)
                       if ((C3 in self.adj_out[C2]) and (C2 in self.adj_out[C3]))
                       ]
        return motifs

    # returns list of tuples of form (C1, C2, C3, C1->C2, C2->C3, C3->C1, C1->C3)
    def directed_cycle_1to3_triad_motifs(self):
        motifs = []
        for C1 in self.hypernodes:
            outgoing = set(self.outgoing_hypernodes(C1))
            for C2, C3 in itertools.permutations(outgoing, 2):
                if C1 in self.adj_out[C2]: continue
                if C2 in self.adj_out[C3]: continue

                if C3 not in self.adj_out[C2]: continue
                if C1 not in self.adj_out[C3]: continue

                motifs += [TriadMotif((C1, C2, C3),
                                      (Hypergraph._sorted_ts(self.adj_out[C1][C2]),
                                       Hypergraph._sorted_ts(self.adj_out[C2][C3]),
                                       Hypergraph._sorted_ts(self.adj_out[C3][C1]),
                                       Hypergraph._sorted_ts(self.adj_out[C1][C3])),
                                      MotifType.DIRECTED_CYCLE_1TO3_TRIADS.name)
                           ]
        # for m in motifs:
        #     print(m)
        return motifs

    # returns list of tuples of form (C1, C2, C3, C1->C2, C2->C1, C1->C3, C3->C1)
    def direciprocal_triad_motifs(self):
        motifs = []
        for C1 in self.hypernodes:
            incoming = set(self.incoming_hypernodes(C1))
            outgoing = set(self.outgoing_hypernodes(C1))
            in_and_out = incoming.intersection(outgoing)
            for C2, C3 in itertools.combinations(in_and_out, 2):
                if C3 in self.adj_out[C2]: continue
                if C2 in self.adj_out[C3]: continue

                motifs += [TriadMotif((C1, C2, C3),
                                      (Hypergraph._sorted_ts(self.adj_out[C1][C2]),
                                       Hypergraph._sorted_ts(self.adj_out[C2][C1]),
                                       Hypergraph._sorted_ts(self.adj_out[C1][C3]),
                                       Hypergraph._sorted_ts(self.adj_out[C3][C1])),
                                      MotifType.DIRECIPROCAL_TRIADS.name)
                           ]
        return motifs

    # returns list of tuples of form (C1, C2, C3, C1->C2, C2->C1, C1->C3, C3->C1, C2->C3)
    def direciprocal_2to3_triad_motifs(self):
        motifs = []
        for C1 in self.hypernodes:
            incoming = set(self.incoming_hypernodes(C1))
            outgoing = set(self.outgoing_hypernodes(C1))
            in_and_out = incoming.intersection(outgoing)
            for C2, C3 in itertools.permutations(in_and_out, 2):
                if C2 in self.adj_out[C3]: continue
                if C3 not in self.adj_out[C2]: continue

                motifs += [TriadMotif((C1, C2, C3),
                                      (Hypergraph._sorted_ts(self.adj_out[C1][C2]),
                                       Hypergraph._sorted_ts(self.adj_out[C2][C1]),
                                       Hypergraph._sorted_ts(self.adj_out[C1][C3]),
                                       Hypergraph._sorted_ts(self.adj_out[C3][C1]),
                                       Hypergraph._sorted_ts(self.adj_out[C2][C3])),
                                      MotifType.DIRECIPROCAL_2TO3_TRIADS.name)
                           ]
        return motifs


    # returns list of tuples of form (C1, C2, C3, C1->C2, C2->C1, C2->C3, C3->C2, C3->C1, C1->C3)
    def trireciprocal_triad_motifs(self):
        # prevents triple-counting
        motifs = []
        for C1, C2, C3 in itertools.combinations(self.hypernodes, 3):
            if C2 not in self.adj_out[C1]: continue
            if C1 not in self.adj_out[C2]: continue
            if C3 not in self.adj_out[C2]: continue
            if C2 not in self.adj_out[C3]: continue
            if C1 not in self.adj_out[C3]: continue
            if C3 not in self.adj_out[C1]: continue

            motifs += [TriadMotif((C1, C2, C3),
                                  (Hypergraph._sorted_ts(self.adj_out[C1][C2]),
                                   Hypergraph._sorted_ts(self.adj_out[C2][C1]),
                                   Hypergraph._sorted_ts(self.adj_out[C2][C3]),
                                   Hypergraph._sorted_ts(self.adj_out[C3][C2]),
                                   Hypergraph._sorted_ts(self.adj_out[C3][C1]),
                                   Hypergraph._sorted_ts(self.adj_out[C1][C3])),
                                  MotifType.TRIRECIPROCAL_TRIADS.name)
                       ]

        return motifs

class HyperConvo(Transformer):
    """
    Encapsulates computation of hypergraph features for a particular
    corpus.

    fit_transform() retrieves features from the corpus conversational
    threads using retrieve_feats, and stores it in the corpus's conversations'
    meta field under the key "hyperconvo"

    Either use the features directly, or use the other transformers, threadEmbedder (https://zissou.infosci.cornell.edu/socialkit/documentation/threadEmbedder.html)
    or communityEmbedder (https://zissou.infosci.cornell.edu/socialkit/documentation/communityEmbedder.html) to embed threads or communities respectively in a low-dimensional
    space for further analysis or visualization.

    As features, we compute the degree distribution statistics from Table 4 of
    http://www.cs.cornell.edu/~cristian/Patterns_of_participant_interactions.html,
    for both a whole conversation and its midthread, and for indegree and
    outdegree distributions of C->C, C->c and c->c edges, as in the paper.
    We also compute the presence and count of each motif type specified in Fig 2.
    However, we do not include features making use of reaction edges, due to our
    inability to release the Facebook data used in the paper (which reaction
    edges are most naturally suited for). In particular, we do not include edge
    distribution statistics from Table 4, as these rely on the presence of
    reaction edges. We hope to implement a more general version of these
    reaction features in an upcoming release.

    :param prefix_len: Length (in number of utterances) of each thread to
            consider when constructing its hypergraph
    :param min_thread_len: Only consider threads of at least this length
    :param include_root: True if root utterance should be included in the utterance thread,
                         False otherwise, i.e. thread begins from top level comment. (Affects prefix_len and min_thread_len counts.)
                         (If include_root is True, then each Conversation will have metadata for one thread, otherwise each Conversation
                         will have metadata for multiple threads - equal to the number of top-level comments.)
    """

    def __init__(self, prefix_len: int=10, min_thread_len: int=10, include_root: bool=True):
        self.prefix_len = prefix_len
        self.min_thread_len = min_thread_len
        self.include_root = include_root

    def transform(self, corpus: Corpus) -> Corpus:
        """
        Same as fit_transform()
        """
        return self.fit_transform(corpus)

    def fit_transform(self, corpus: Corpus) -> Corpus:
        """
        fit_transform() retrieves features from the corpus conversational
        threads using retrieve_feats()

        :param corpus: Corpus object to retrieve feature information from

        :return: corpus with conversations having a new meta field "hyperconvo" containing the stats generated by retrieve_feats(). Each conversation's metadata then contains the stats for the thread(s) it contains.
        """
        feats = HyperConvo.retrieve_feats(corpus,
                                          prefix_len=self.prefix_len,
                                          min_thread_len=self.min_thread_len,
                                          include_root=self.include_root)
        if self.include_root: # threads start at root (post)
            for root_id in feats.keys():
                convo = corpus.get_conversation(root_id)
                convo.add_meta("hyperconvo", {root_id: feats[root_id]})
        else: # threads start at top-level-comment
            # Construct top-level-comment to root mapping
            threads = corpus.utterance_threads(prefix_len=self.prefix_len, include_root=False)

            root_to_tlc = dict()
            for tlc_id, utts in threads.items():
                thread_root = threads[tlc_id][tlc_id].root
                if thread_root in root_to_tlc:
                    root_to_tlc[thread_root][tlc_id] = feats[tlc_id]
                else:
                    root_to_tlc[thread_root] = {tlc_id: feats[tlc_id]}

            for root_id in root_to_tlc:
                convo = corpus.get_conversation(root_id)
                convo.add_meta("hyperconvo", root_to_tlc[root_id])

        return corpus

    @staticmethod
    def _make_hypergraph(corpus: Optional[Corpus]=None,
                         uts: Optional[Dict[Hashable, Utterance]]=None,
                         exclude_id: Hashable=None) -> Hypergraph:
        """
        Construct a Hypergraph from all the utterances of a Corpus, or a specified subset of utterances

        :param corpus: A Corpus to extract utterances from
        :param uts: Subset of utterances to construct a Hypergraph from
        :param exclude_id: id of utterance to exclude from Hypergraph construction

        :return: Hypergraph object
        """
        if uts is None:
            if corpus is None:
                raise RuntimeError("fit_transform() helper method _make_hypergraph()"
                                   "has no valid corpus / utterances input")
            uts = {utt.id: utt for utt in corpus.iter_utterances()}

        G = Hypergraph()
        username_to_utt_ids = dict()
        reply_edges = []
        speaker_to_reply_tos = defaultdict(list)
        speaker_target_pairs = set()
        # nodes
        for ut in sorted(uts.values(), key=lambda h: h.get("timestamp")):
            if ut.get("id") != exclude_id:
                if ut.get("user") not in username_to_utt_ids:
                    username_to_utt_ids[ut.get("user")] = set()
                username_to_utt_ids[ut.get("user")].add(ut.get("id"))
                if ut.get("reply_to") is not None and ut.get("reply_to") in uts \
                        and ut.get("reply_to") != exclude_id:
                    reply_edges.append((ut.get("id"), ut.get("reply_to")))
                    speaker_to_reply_tos[ut.user].append(ut.get("reply_to"))
                    speaker_target_pairs.add((ut.user, uts[ut.reply_to].user, ut.timestamp, ut.text, ut.reply_to, ut.root))
                G.add_node(ut.get("id"), info=ut.__dict__)
        # hypernodes
        for u, ids in username_to_utt_ids.items():
            G.add_hypernode(u, ids, info=u.meta)
        # reply edges
        for u, v in reply_edges:
            # print("ADDING TIMESTAMP")
            G.add_edge(u, v)
        # user to utterance response edges
        for u, reply_tos in speaker_to_reply_tos.items():
            for reply_to in reply_tos:
                G.add_edge(u, reply_to)
        # user to user response edges
        for u, v, timestamp, text, reply_to, top_level_comment in speaker_target_pairs:
            G.add_edge(u, v, {'timestamp': timestamp,
                              'text': text,
                              'speaker': u,
                              'target': v,
                              'reply_to': reply_to,
                              'top_level_comment': top_level_comment,
                              'root': (reply_to == top_level_comment)
                              })
        return G

    @staticmethod
    def _node_type_name(b: bool) -> str:
        """
        Helper method to get node type name (C or c)

        :param b: Bool, where True indicates node is a Hypernode
        :return: "C" if True, "c" if False
        """
        return "C" if b else "c"

    @staticmethod
    def _degree_feats(uts: Optional[Dict[Hashable, Utterance]]=None,
                      G: Optional[Hypergraph]=None,
                      name_ext: str="",
                      exclude_id: Optional[Hashable]=None) -> Dict:
        """
        Helper method for retrieve_feats().
        Generate statistics on degree-related features in a Hypergraph (G), or a Hypergraph
        constructed from provided utterances (uts)

        :param uts: utterances to construct Hypergraph from
        :param G: Hypergraph to calculate degree features statistics from
        :param name_ext: Suffix to append to feature name
        :param exclude_id: id of utterance to exclude from Hypergraph construction
        :return: A stats dictionary, i.e. a dictionary of feature names to feature values. For degree-related features specifically.
        """
        assert uts is None or G is None
        if G is None:
            G = HyperConvo._make_hypergraph(uts, exclude_id=exclude_id)

        stat_funcs = {
            "max": np.max,
            "argmax": np.argmax,
            "norm.max": lambda l: np.max(l) / np.sum(l),
            "2nd-largest": lambda l: np.partition(l, -2)[-2] if len(l) > 1
            else np.nan,
            "2nd-argmax": lambda l: (-l).argsort()[1] if len(l) > 1 else np.nan,
            "norm.2nd-largest": lambda l: np.partition(l, -2)[-2] / np.sum(l)
            if len(l) > 1 else np.nan,
            "mean": np.mean,
            "mean-nonzero": lambda l: np.mean(l[l != 0]),
            "prop-nonzero": lambda l: np.mean(l != 0),
            "prop-multiple": lambda l: np.mean(l[l != 0] > 1),
            "entropy": scipy.stats.entropy,
            "2nd-largest / max": lambda l: np.partition(l, -2)[-2] / np.max(l)
            if len(l) > 1 else np.nan
        }

        stats = {}
        for from_hyper in [False, True]:
            for to_hyper in [False, True]:
                if not from_hyper and to_hyper: continue  # skip c -> C
                outdegrees = np.array(G.outdegrees(from_hyper, to_hyper))
                indegrees = np.array(G.indegrees(from_hyper, to_hyper))

                for stat, stat_func in stat_funcs.items():
                    stats["{}[outdegree over {}->{} {}responses]".format(stat,
                                                                         HyperConvo._node_type_name(from_hyper),
                                                                         HyperConvo._node_type_name(to_hyper),
                                                                         name_ext)] = stat_func(outdegrees)
                    stats["{}[indegree over {}->{} {}responses]".format(stat,
                                                                        HyperConvo._node_type_name(from_hyper),
                                                                        HyperConvo._node_type_name(to_hyper),
                                                                        name_ext)] = stat_func(indegrees)
        return stats

    @staticmethod
    def probabilities(transitions: Dict):
        """
        Takes a transitions count dictionary Dict[(MotifType.name->MotifType.name)->Int]
        :return: transitions probability dictionary Dict[(MotifType.name->MotifType.name)->Float]
        """
        probs = dict()

        for parent, children in TriadMotif.relations().items():
            total = sum(transitions[(parent, c)] for c in children) + transitions[(parent, parent)]
            probs[(parent, parent)] = (transitions[(parent, parent)] / total) if total > 0 else 0
            for c in children:
                probs[(parent, c)] = (transitions[(parent, c)] / total) if total > 0 else 0

        return probs

    @staticmethod
    def _latent_motif_count(motifs, trans: bool):
        """
        Takes a dictionary of (MotifType.name, List[Motif]) and a bool prob, indicating whether
        transition probabilities need to be returned
        :return: Returns a tuple of a dictionary of latent motif counts
        and a dictionary of motif->motif transition probabilities
         (Dict[MotifType.name->Int], Dict[(MotifType.name->MotifType.name)->Float])
         The second element is None if prob=False
        """
        latent_motif_count = {motif_type.name: 0 for motif_type in MotifType}

        transitions = TriadMotif.transitions()
        for motif_type, motif_instances in motifs.items():
            for motif_instance in motif_instances:
                curr_motif = motif_instance
                child_motif_type = curr_motif.get_type()
                # Reflexive edge
                transitions[(child_motif_type, child_motif_type)] += 1

                # print(transitions)
                while True:
                    latent_motif_count[curr_motif.get_type()] +=  1
                    curr_motif = curr_motif.regress()
                    if curr_motif is None: break
                    parent_motif_type = curr_motif.get_type()
                    transitions[(parent_motif_type, child_motif_type)] += 1
                    child_motif_type = parent_motif_type

        return latent_motif_count, transitions

    @staticmethod
    def _motif_feats(uts: Optional[Dict[Hashable, Utterance]]=None,
                     G: Hypergraph=None,
                     name_ext: str="",
                     exclude_id: str=None,
                     latent=True,
                     trans=True) -> Dict:
        """
        Helper method for retrieve_feats().
        Generate statistics on degree-related features in a Hypergraph (G), or a Hypergraph
        constructed from provided utterances (uts)
        :param uts: utterances to construct Hypergraph from
        :param G: Hypergraph to calculate degree features statistics from
        :param name_ext: Suffix to append to feature name
        :param exclude_id: id of utterance to exclude from Hypergraph construction
        :return: A dictionary from a thread root id to its stats dictionary, which is a dictionary from feature names to feature values. For motif-related features specifically.
        """
        assert uts is None or G is None
        if G is None:
            G = HyperConvo._make_hypergraph(uts=uts, exclude_id=exclude_id)

        motifs = G.extract_motifs()

        stat_funcs = {
            # "is-present": lambda l: len(l) > 0,
            "count": len
        }

        stats = {}

        for motif_type in motifs:
            for stat, stat_func in stat_funcs.items():
                stats["{}[{}{}]".format(stat, str(motif_type), name_ext)] = \
                    stat_func(motifs[motif_type])

        if latent:
            latent_motif_count, transitions = HyperConvo._latent_motif_count(motifs, trans=trans)
            for motif_type in latent_motif_count:
                # stats["is-present[LATENT_{}{}]".format(motif_type, name_ext)] = \
                #     (latent_motif_count[motif_type] > 0)
                stats["count[LATENT_{}{}]".format(motif_type, name_ext)] = latent_motif_count[motif_type]

            if trans:
                assert transitions is not None
                for p, v in transitions.items():
                    stats["trans[{}]".format(p)] = v

        return stats


    @staticmethod
    def retrieve_texts(corpus: Corpus, prefix_len=10, min_thread_len=10):
        threads_motifs = {}
        for i, (root, thread) in enumerate(
                corpus.utterance_threads(prefix_len=prefix_len).items()):
            if len(thread) < min_thread_len: continue

            G = HyperConvo._make_hypergraph(uts=thread)
            motifs = G.extract_motifs()

            motifs_texts = dict()
            for motif_type in motifs:
                motifs_texts[motif_type] = [motif.get_text() for motif in motifs[motif_type]]

            threads_motifs[root] = motifs_texts
        return threads_motifs

    @staticmethod
    def retrieve_motifs(corpus: Corpus, prefix_len=10, min_thread_len=10):
        threads_motifs = {}
        for i, (root, thread) in enumerate(
                corpus.utterance_threads(prefix_len=prefix_len).items()):
            if len(thread) < min_thread_len: continue
            G = HyperConvo._make_hypergraph(uts=thread)
            motifs = G.extract_motifs()
            threads_motifs[root] = motifs

        return threads_motifs

    @staticmethod
    def get_threads_motifs(corpus: Corpus, prefix_len=10, min_thread_len=10):
        threads_motifs = {}

        for i, (root, thread) in enumerate(
                corpus.utterance_threads(prefix_len=prefix_len).items()):
            if len(thread) < min_thread_len: continue
            G = HyperConvo._make_hypergraph(uts=thread)

            threads_motifs[root] = G.extract_motifs()
        return threads_motifs


    @staticmethod
    def retrieve_feats(corpus: Corpus, prefix_len: int=10,
                       min_thread_len: int=10,
                       include_root: bool=True) -> Dict[Hashable, Dict]:
        """
        Retrieve all hypergraph features for a given corpus (viewed as a set
        of conversation threads).

        See init() for further documentation.

        :return: A dictionary from a thread root id to its stats dictionary,
            which is a dictionary from feature names to feature values. For degree-related
            features specifically.
        """

        threads_stats = dict()

        for i, (root, thread) in enumerate(
                corpus.utterance_threads(prefix_len=prefix_len, include_root=include_root).items()):
            if len(thread) < min_thread_len: continue
            stats = {}
            G = HyperConvo._make_hypergraph(uts=thread)
            G_mid = HyperConvo._make_hypergraph(uts=thread, exclude_id=root)
            for k, v in HyperConvo._degree_feats(G=G).items(): stats[k] = v
            for k, v in HyperConvo._motif_feats(G=G).items(): stats[k] = v
            for k, v in HyperConvo._degree_feats(G=G_mid,
                                           name_ext="mid-thread ").items(): stats[k] = v
            for k, v in HyperConvo._motif_feats(G=G_mid,
                                          name_ext=" over mid-thread").items(): stats[k] = v
            threads_stats[root] = stats

        return threads_stats

