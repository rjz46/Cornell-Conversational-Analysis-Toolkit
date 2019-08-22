import itertools
from typing import Tuple, List, Dict, Optional, Hashable, Collection

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

    def reciprocity_motifs(self) -> List[Tuple]:
        """
        :return: List of tuples of form (C1, c1, c2, C1->c2, c2->c1) as in paper
        """
        motifs = []
        for C1, c1_nodes in self.hypernodes.items():
            for c1 in c1_nodes:
                motifs += [(C1, c1, c2, e1, e2) for c2 in self.adj_in[c1] if
                           c2 in self.nodes and c2 in self.adj_out[C1]
                           for e1 in self.adj_out[C1][c2]
                           for e2 in self.adj_out[c2][c1]]
        return motifs

    def external_reciprocity_motifs(self) -> List[Tuple]:
        """
        :return: List of tuples of form (C3, c2, c1, C3->c2, c2->c1) as in paper
        """
        motifs = []
        for C3 in self.hypernodes:
            for c2 in self.adj_out[C3]:
                if c2 in self.nodes:
                    motifs += [(C3, c2, c1, e1, e2) for c1 in
                               set(self.adj_out[c2].keys()) - self.hypernodes[C3]
                               if c1 in self.nodes
                               for e1 in self.adj_out[C3][c2]
                               for e2 in self.adj_out[c2][c1]]
        return motifs

    def dyadic_interaction_motifs(self) -> List[Tuple]:
        """
        :return: List of tuples of form (C1, C2, C1->C2, C2->C1) as in paper
        """

        motifs = []
        for C1 in self.hypernodes:
            motifs += [(C1, C2, e1, e2) for C2 in self.adj_out[C1] if C2 in
                       self.hypernodes and C1 in self.adj_out[C2]
                       for e1 in self.adj_out[C1][C2]
                       for e2 in self.adj_out[C2][C1]]
        return motifs

    def incoming_triad_motifs(self) -> List[Tuple]:
        """
        :return: List of tuples of form (C1, C2, C3, C2->C1, C3->C1) as in paper
        """
        motifs = []
        for C1 in self.hypernodes:
            incoming = list(self.adj_in[C1].keys())
            motifs += [(C1, C2, C3, e1, e2) for C2, C3 in
                       itertools.combinations(incoming, 2)
                       for e1 in self.adj_out[C2][C1]
                       for e2 in self.adj_out[C3][C1]]
        return motifs

    def outgoing_triad_motifs(self) -> List[Tuple]:
        """
        :return: List of tuples of form (C1, C2, C3, C1->C2, C1->C3) as in paper
        """
        motifs = []
        for C1 in self.hypernodes:
            outgoing = list(self.adj_out[C1].keys())
            motifs += [(C1, C2, C3, e1, e2) for C2, C3 in
                       itertools.combinations(outgoing, 2)
                       for e1 in self.adj_out[C1][C2]
                       for e2 in self.adj_out[C1][C3]]
        return motifs
