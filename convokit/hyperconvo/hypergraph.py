import itertools
from typing import Tuple, List, Dict, Optional, Hashable, Collection
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

