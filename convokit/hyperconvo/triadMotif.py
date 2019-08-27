from enum import Enum, auto
from typing import Tuple, Dict, List, Optional
from graphviz import Digraph


class MotifType(Enum):
    NO_EDGE_TRIADS = auto()
    SINGLE_EDGE_TRIADS = auto()
    INCOMING_TRIADS = auto()
    OUTGOING_TRIADS = auto()
    DYADIC_TRIADS = auto()
    UNIDIRECTIONAL_TRIADS = auto()
    INCOMING_2TO3_TRIADS = auto()
    INCOMING_1TO3_TRIADS = auto()
    DIRECTED_CYCLE_TRIADS = auto()
    OUTGOING_3TO1_TRIADS = auto()
    INCOMING_RECIPROCAL_TRIADS = auto()
    OUTGOING_RECIPROCAL_TRIADS = auto()
    DIRECTED_CYCLE_1TO3_TRIADS = auto()
    DIRECIPROCAL_TRIADS = auto()
    DIRECIPROCAL_2TO3_TRIADS = auto()
    TRIRECIPROCAL_TRIADS = auto()


class TriadMotif:
    """
    Represents a triadic motif, consisting of three hypernodes and directed edges between the hypernodes
    Contains functionality to temporally regress a motif to its antecedent stages
    """
    def __init__(self, hypernodes: Tuple, edges: Tuple[List[Dict[str, int]], ...], triad_type: str):
        self.hypernodes = hypernodes
        self.edges = edges
        self.triad_type = triad_type
        self.labels = TriadMotif.edge_labels()[triad_type]

    def get_hypernodes(self):
        return self.hypernodes

    def get_edges(self):
        return self.edges

    def get_type(self):
        return self.triad_type

    def _last_added_edge_idx(self):
        edges = self.edges
        max_idx = 0
        max_timestamp = 0
        # print("The current motif type is: {}".format(self.type))
        # print("This is what the edge set looks like: {}".format(edges))
        for i, edge_list in enumerate(edges):
            timestamp = edge_list[0]['timestamp'] # only need to get first edge of type to determine how to regress
            if timestamp >= max_timestamp:
                max_idx = i
                max_timestamp = timestamp
        return max_idx

    def get_text(self):
        texts = []
        for e in self.edges:
            texts.append(e[0]["text"])
        return texts


    @staticmethod
    def edge_labels():
        """
        Returns a Dict[MotifType -> List of edges constituting motif]
        :return:
        """
        return {
            MotifType.NO_EDGE_TRIADS.name: [],
            MotifType.SINGLE_EDGE_TRIADS.name: ["C2->C1"],
            MotifType.INCOMING_TRIADS.name: ["C2->C1", "C3->C1"],
            MotifType.OUTGOING_TRIADS.name: ["C1->C2", "C1->C3"],
            MotifType.DYADIC_TRIADS.name: ["C1->C2", "C2->C1"],
            MotifType.UNIDIRECTIONAL_TRIADS.name: ["C2->C1", "C3->C2"],

            # (C1, C2, C3, C2->C1, C3->C1, C2->C3)
            MotifType.INCOMING_2TO3_TRIADS.name: ["C2->C1", "C3->C1", "C2->C3"],

            # (C1, C2, C3, C1->C2, C2->C3, C3->C1)
            MotifType.DIRECTED_CYCLE_TRIADS.name: ["C1->C2", "C2->C3", "C3->C1"],

            # (C1, C2, C3, C2->C1, C3->C1, C1->C3)
            MotifType.INCOMING_1TO3_TRIADS.name: ["C2->C1", "C3->C1", "C1->C3"],

            # (C1, C2, C3, C1->C2, C1->C3, C3->C1)
            MotifType.OUTGOING_3TO1_TRIADS.name: ["C1->C2", "C1->C3", "C3->C1"],

            # (C1, C2, C3, C2->C1, C3->C1, C2->C3, C3->C2)
            MotifType.INCOMING_RECIPROCAL_TRIADS.name: ["C2->C1", "C3->C1", "C2->C3", "C3->C2"],

            # (C1, C2, C3, C1->C2, C1->C3, C2->C3, C3->C2)
            MotifType.OUTGOING_RECIPROCAL_TRIADS.name: ["C1->C2", "C1->C3", "C2->C3", "C3->C2"],

            # (C1, C2, C3, C1->C2, C2->C3, C3->C1, C1->C3)
            MotifType.DIRECTED_CYCLE_1TO3_TRIADS.name: ["C1->C2", "C2->C3", "C3->C1", "C1->C3"],

            # (C1, C2, C3, C1->C2, C2->C1, C1->C3, C3->C1)
            MotifType.DIRECIPROCAL_TRIADS.name: ["C1->C2", "C2->C1", "C1->C3", "C3->C1"],

            # (C1, C2, C3, C1->C2, C1->C3, C2->C1, C3->C1, C2->C3) wrong
            # (C1, C2, C3, C1->C2, C2->C1, C1->C3, C3->C1, C2->C3)
            MotifType.DIRECIPROCAL_2TO3_TRIADS.name: ["C1->C2", "C2->C1", "C1->C3", "C3->C1", "C2->C3"],

            # (C1, C2, C3, C1->C2, C2->C1, C2->C3, C3->C2, C3->C1, C1->C3)}
            MotifType.TRIRECIPROCAL_TRIADS.name: ["C1->C2", "C2->C1", "C2->C3", "C3->C2", "C3->C1", "C1->C3"]
        }

    @staticmethod
    def get_motif_types() -> List[str]:
        """
        :return: List of names of Motifs
        """
        return list(TriadMotif.edge_labels())

    # delete last edge and reorder remaining edges + hypernodes to conform to antecedent motif edge ordering
    def delete_and_reorder(self, idx_to_delete: int) -> Tuple[Tuple, Tuple]:
        assert self.triad_type != MotifType.NO_EDGE_TRIADS.name

        h = self.hypernodes
        #remaining edges
        e = self.edges[:idx_to_delete] + self.edges[idx_to_delete+1:]

        if self.triad_type == MotifType.SINGLE_EDGE_TRIADS.name:
            return h, e

        elif self.triad_type == MotifType.INCOMING_TRIADS.name: # (C1, C2, C3, C2->C1, C3->C1)
            if idx_to_delete == 0: # becomes single edge: (C1, C2, C3, C1->C2)
                return (h[2], h[0], h[1]), e
            elif idx_to_delete == 1:
                return (h[1], h[0], h[2]), e
            else:
                raise Exception("Invalid regression index: {} for triad type: {}".format(int, self.triad_type))

        elif self.triad_type == MotifType.OUTGOING_TRIADS.name: # (C1, C2, C3, C1->C2, C1->C3)
            if idx_to_delete == 0: # becomes single edge: (C1, C2, C3, C1->C2)
                return (h[0], h[2], h[1]), e
            elif idx_to_delete == 1:
                return h, e
            else:
                raise Exception("Invalid regression index: {} for triad type: {}".format(int, self.triad_type))

        elif self.triad_type == MotifType.DYADIC_TRIADS.name: # (C1, C2, C3, C1->C2, C2->C1)
            if idx_to_delete == 0: # becomes single edge: (C1, C2, C3, C1->C2)
                return (h[1], h[0], h[2]), e
            elif idx_to_delete == 1:
                return h, e
            else:
                raise Exception("Invalid regression index: {} for triad type: {}".format(int, self.triad_type))

        elif self.triad_type == MotifType.UNIDIRECTIONAL_TRIADS.name: # (C1, C2, C3, C1->C2, C2->C3)
            if idx_to_delete == 0: # becomes single edge: (C1, C2, C3, C1->C2)
                return (h[1], h[2], h[0]), e
            elif idx_to_delete == 1:
                return h, e
            else:
                raise Exception("Invalid regression index: {} for triad type: {}".format(int, self.triad_type))

        elif self.triad_type == MotifType.INCOMING_2TO3_TRIADS.name: # (C1, C2, C3, C2->C1, C3->C1, C2->C3)
            if idx_to_delete == 0: # becomes unidirectional: (C1, C2, C3, C1->C2, C2->C3)
                return (h[1], h[2], h[0]), (e[1], e[0])
            elif idx_to_delete == 1: # becomes outgoing: (C1, C2, C3, C1->C2, C1->C3)
                return (h[1], h[0], h[2]), e
            elif idx_to_delete == 2: # becomes incoming (C1, C2, C3, C2->C1, C3->C1)
                return h, e
            else:
                raise Exception("Invalid regression index: {} for triad type: {}".format(int, self.triad_type))

        elif self.triad_type == MotifType.DIRECTED_CYCLE_TRIADS.name: # (C1, C2, C3, C1->C2, C2->C3, C3->C1)
            if idx_to_delete == 0: # becomes unidirectional: (C1, C2, C3, C1->C2, C2->C3)
                return (h[1], h[2], h[0]), e
            elif idx_to_delete == 1: # becomes unidirectional: (C1, C2, C3, C1->C2, C2->C3)
                return (h[2], h[0], h[1]), (e[1], e[0])
            elif idx_to_delete == 2: # becomes unidirectional: (C1, C2, C3, C1->C2, C2->C3)
                return h, e
            else:
                raise Exception("Invalid regression index: {} for triad type: {}".format(int, self.triad_type))

        elif self.triad_type == MotifType.INCOMING_1TO3_TRIADS.name: # (C1, C2, C3, C2->C1, C3->C1, C1->C3)
            if idx_to_delete == 0: # becomes dyadic: (C1, C2, C3, C1->C2, C2->C1)
                return (h[2], h[0], h[1]), e
            elif idx_to_delete == 1: # becomes unidirectional: (C1, C2, C3, C1->C2, C2->C3)
                return (h[1], h[0], h[2]), e
            elif idx_to_delete == 2: # becomes incoming: (C1, C2, C3, C2->C1, C3->C1)
                return h, e
            else:
                raise Exception("Invalid regression index: {} for triad type: {}".format(int, self.triad_type))

        elif self.triad_type == MotifType.OUTGOING_3TO1_TRIADS.name: # (C1, C2, C3, C1->C2, C1->C3, C3->C1)
            if idx_to_delete == 0: # becomes dyadic: (C1, C2, C3, C1->C2, C2->C1)
                return (h[0], h[2], h[1]), e
            elif idx_to_delete == 1: # becomes unidirectional: (C1, C2, C3, C1->C2, C2->C3)
                return (h[2], h[0], h[1]), (e[1], e[0])
            elif idx_to_delete == 2: # becomes outgoing: (C1, C2, C3, C1->C2, C1->C3)
                return h, e
            else:
                raise Exception("Invalid regression index: {} for triad type: {}".format(int, self.triad_type))

        elif self.triad_type == MotifType.INCOMING_RECIPROCAL_TRIADS.name: # (C1, C2, C3, C2->C1, C3->C1, C2->C3, C3->C2)
            if idx_to_delete == 0: # becomes outgoing_3to1: (C1, C2, C3, C1->C2, C1->C3, C3->C1)
                return (h[2], h[0], h[1]), (e[0], e[2], e[1])
            elif idx_to_delete == 1: # becomes outgoing_3to1: (C1, C2, C3, C1->C2, C1->C3, C3->C1)
                return (h[1], h[0], h[2]), e
            elif idx_to_delete == 2: # becomes incoming_2to3: (C1, C2, C3, C2->C1, C3->C1, C2->C3)
                return (h[0], h[2], h[1]), e
            elif idx_to_delete == 3: # becomes incoming_2to3: (C1, C2, C3, C2->C1, C3->C1, C2->C3)
                return h, e
            else:
                raise Exception("Invalid regression index: {} for triad type: {}".format(int, self.triad_type))


        elif self.triad_type == MotifType.OUTGOING_RECIPROCAL_TRIADS.name: # (C1, C2, C3, C1->C2, C1->C3, C2->C3, C3->C2)
            if idx_to_delete == 0: # becomes incoming_1to3: (C1, C2, C3, C2->C1, C3->C1, C1->C3)
                return (h[2], h[0], h[1]), e
            elif idx_to_delete == 1: # becomes incoming_1to3: (C1, C2, C3, C2->C1, C3->C1, C1->C3)
                #C1->C2,  C3->C2, C2->C3 (after rearrange)
                return (h[1], h[0], h[2]), (e[0], e[2], e[1])
            elif idx_to_delete == 2: # becomes incoming_2to3: (C1, C2, C3, C2->C1, C3->C1, C2->C3)
                # C1->C2, C3->C2, C1->C3 (after rearrange)
                return  (h[1], h[0], h[2]), (e[0], e[2], e[1])
            elif idx_to_delete == 3: # becomes incoming_2to3: (C1, C2, C3, C2->C1, C3->C1, C2->C3)
                # C1->C2, C3->C2, C1->C3
                return (h[1], h[0], h[2]), (e[0], e[2], e[1])
            elif idx_to_delete == 4: # becomes incoming_2to3: (C1, C2, C3, C2->C1, C3->C1, C2->C3)
                # C1->C3, C2->C3, C1->C2  (after rearrange)
                return (h[2], h[0], h[1]), (e[1], e[2], e[0])
            else:
                raise Exception("Invalid regression index: {} for triad type: {}".format(int, self.triad_type))

        elif self.triad_type == MotifType.DIRECTED_CYCLE_1TO3_TRIADS.name: # (C1, C2, C3, C1->C2, C2->C3, C3->C1, C1->C3)
            if idx_to_delete == 0: # becomes incoming_1to3: (C1, C2, C3, C2->C1, C3->C1, C1->C3)
                # C2->C3, C3->C1, C1->C3
                return (h[2], h[1], h[0]), (e[0], e[2], e[1])
            elif idx_to_delete == 1: # becomes outgoing_3to1: (C1, C2, C3, C1->C2, C1->C3, C3->C1)
                # C1->C2, C3->C1, C1->C3
                return h, (e[0], e[2], e[1])
            elif idx_to_delete == 2: # becomes incoming_2to3: (C1, C2, C3, C2->C1, C3->C1, C2->C3)
                # C2->C3, C1->C3, C1->C2 (rearrange)
                return (h[2], h[1], h[0]), (e[1], e[2], e[0])
            elif idx_to_delete == 3: # becomes directed cycle: (C1, C2, C3, C1->C2, C2->C3, C3->C1)
                # C1->C2, C2->C3, C3->C1
                return h, e
            else:
                raise Exception("Invalid regression index: {} for triad type: {}".format(int, self.triad_type))

        elif self.triad_type == MotifType.DIRECIPROCAL_TRIADS.name: # (C1, C2, C3, C1->C2, C2->C1, C1->C3, C3->C1)
            # C2->C1, C1->C3, C3->C1
            if idx_to_delete == 0: # becomes incoming_1to3: (C1, C2, C3, C2->C1, C3->C1, C1->C3)
                return h, (e[0], e[2], e[1])
            elif idx_to_delete == 1: # becomes outgoing_3to1: (C1, C2, C3, C1->C2, C1->C3, C3->C1)
                return h, e
            elif idx_to_delete == 2: # becomes incoming_1to3: (C1, C2, C3, C2->C1, C3->C1, C1->C3)
                # C1->C2, C2->C1, C3->C1
                # C3->C1, C2->C1, C1->C2 (rearrange)
                return (h[0], h[2], h[1]), (e[2], e[1], e[0])
            elif idx_to_delete == 3: # becomes outgoing_3to1: (C1, C2, C3, C1->C2, C1->C3, C3->C1)
                # C1->C2, C2->C1, C1->C3
                # C1->C3, C1->C2, C2->C1 (rearrange)
                return (h[0], h[2], h[1]), (e[2], e[0], e[1])
            else:
                raise Exception("Invalid regression index: {} for triad type: {}".format(int, self.triad_type))


        elif self.triad_type == MotifType.DIRECIPROCAL_2TO3_TRIADS.name: # (C1, C2, C3, C1->C2, C2->C1, C1->C3, C3->C1, C2->C3)
            if idx_to_delete == 0: # becomes outgoing_reciprocal: (C1, C2, C3, C1->C2, C1->C3, C2->C3, C3->C2)
                # C2->C1, C1->C3, C3->C1, C2->C3 (deleted)
                # C2->C1, C2->C3, C1->C3, C3->C1 (rearrange)
                return h, (e[0], e[3], e[1], e[2])
            elif idx_to_delete == 1: # becomes directed_cycle_1to3: (C1, C2, C3, C1->C2, C2->C3, C3->C1, C1->C3)
                # C1->C2, C1->C3, C3->C1, C2->C3 (deleted)
                # C1->C2, C2->C3, C3->C1, C1->C3 (rearrange)
                return h, (e[0], e[3], e[2], e[1])
            elif idx_to_delete == 2: # becomes directed_cycle_1to3: (C1, C2, C3, C1->C2, C2->C3, C3->C1, C1->C3)
                # C1->C2, C2->C1, C3->C1, C2->C3 (deleted)
                # C2->C3, C3->C1, C1->C2, C2->C1 (rearranged)
                # C1->C2, C2->C3, C3->C1, C1->C3 (transposed)
                return (h[1], h[2], h[0]), (e[3], e[2], e[0], e[1])
            elif idx_to_delete == 3: # becomes incoming_reciprocal: (C1, C2, C3, C2->C1, C3->C1, C2->C3, C3->C2)
                # C1->C2, C2->C1, C1->C3, C2->C3
                # C1->C3, C2->C3, C1->C2, C2->C1 (rearrange)
                return (h[2], h[0], h[1]), (e[2], e[3], e[0], e[1])
            elif idx_to_delete == 4: # becomes direciprocal: (C1, C2, C3, C1->C2, C2->C1, C1->C3, C3->C1)
                return h, e
            else:
                raise Exception("Invalid regression index: {} for triad type: {}".format(int, self.triad_type))

        elif self.triad_type == MotifType.TRIRECIPROCAL_TRIADS.name: # (C1, C2, C3, C1->C2, C2->C1, C2->C3, C3->C2, C3->C1, C1->C3)
            # regress to direciprocal_2to3 (C1, C2, C3, C1->C2, C1->C3, C2->C1, C3->C1, C2->C3)
            if idx_to_delete == 0:
                # C2->C1, C2->C3, C3->C2, C3->C1, C1->C3 (deleted)
                # C3->C2, C3->C1, C2->C3, C1->C3, C2->C1 (rearrange)
                return (h[2], h[1], h[0]), (e[2], e[3], e[1], e[4], e[0])

            elif idx_to_delete == 1: # C1->C2, C1->C3, C2->C1, C3->C1, C2->C3 (desired)
                # C1->C2, C2->C3, C3->C2, C3->C1, C1->C3 (deleted)
                # C3->C1, C3->C2, C1->C3, C2->C3, C1->C2 (rearranged)
                return (h[2], h[0], h[1]), (e[3], e[2], e[4], e[1], e[0])

            elif idx_to_delete == 2: # C1->C2, C1->C3, C2->C1, C3->C1, C2->C3
                # C1->C2, C2->C1, C3->C2, C3->C1, C1->C3 (deleted)
                # C1->C3, C1->C2, C3->C1, C2->C1, C3->C2
                return (h[0], h[2], h[1]), (e[4], e[0], e[3], e[1], e[2])

            elif idx_to_delete == 3: # C1->C2, C1->C3, C2->C1, C3->C1, C2->C3
                # C1->C2, C2->C1, C2->C3, C3->C1, C1->C3 (deleted)
                return h, (e[0], e[4], e[1], e[3], e[2])

            elif idx_to_delete == 4: # C1->C2, C1->C3, C2->C1, C3->C1, C2->C3
                # C1->C2, C2->C1, C2->C3, C3->C2, C1->C3 (deleted)
                # C2->C1, C2->C3, C1->C2, C3->C2, C1->C3 (rearranged)
                return (h[1], h[0], h[2]), (e[1], e[2], e[0], e[3], e[4])

            elif idx_to_delete == 5: # C1->C2, C1->C3, C2->C1, C3->C1, C2->C3
                # C1->C2, C2->C1, C2->C3, C3->C2, C3->C1 (deleted)
                # C2->C3, C2->C1, C3->C2, C1->C2, C3->C1 (rearranged)
                return (h[1], h[2], h[0]), (e[2], e[1], e[3], e[0], e[4])

            else:
                raise Exception("Invalid regression index: {} for triad type: {}".format(int, self.triad_type))

        else:
            raise Exception("No such motif type: {}".format(self.triad_type))

    # returns a motif with the last edge removed
    def regress(self,verbose=False):
        if self.triad_type == MotifType.NO_EDGE_TRIADS.name: return None
        last_edge_idx = self._last_added_edge_idx()
        if verbose: print("Last edge index is: {}".format(last_edge_idx))
        hypernodes, edges = self.delete_and_reorder(last_edge_idx)

        new_type = TriadMotif.regression()[self.triad_type][last_edge_idx]

        return TriadMotif(hypernodes, edges, new_type)

    @staticmethod
    def regression():
        """
        :return: dictionary where key is MotifType, and value is dictionary of motif types that result from deletion
        # of specified edge number
        """
        return {
            MotifType.NO_EDGE_TRIADS.name: {},

            MotifType.SINGLE_EDGE_TRIADS.name: {0: MotifType.NO_EDGE_TRIADS.name},

            MotifType.INCOMING_TRIADS.name: {0: MotifType.SINGLE_EDGE_TRIADS.name, 1: MotifType.SINGLE_EDGE_TRIADS.name},
            MotifType.OUTGOING_TRIADS.name: {0: MotifType.SINGLE_EDGE_TRIADS.name, 1: MotifType.SINGLE_EDGE_TRIADS.name},
            MotifType.DYADIC_TRIADS.name: {0: MotifType.SINGLE_EDGE_TRIADS.name, 1: MotifType.SINGLE_EDGE_TRIADS.name},
            MotifType.UNIDIRECTIONAL_TRIADS.name: {0: MotifType.SINGLE_EDGE_TRIADS.name, 1: MotifType.SINGLE_EDGE_TRIADS.name},

            # (C1, C2, C3, C2->C1, C3->C1, C2->C3)
            MotifType.INCOMING_2TO3_TRIADS.name: {0: MotifType.UNIDIRECTIONAL_TRIADS.name,
                                                  1: MotifType.OUTGOING_TRIADS.name,
                                                  2: MotifType.INCOMING_TRIADS.name},

            # (C1, C2, C3, C1->C2, C2->C3, C3->C1)
            MotifType.DIRECTED_CYCLE_TRIADS.name: {0: MotifType.UNIDIRECTIONAL_TRIADS.name,
                                                   1: MotifType.UNIDIRECTIONAL_TRIADS.name,
                                                   2: MotifType.UNIDIRECTIONAL_TRIADS.name},

            # (C1, C2, C3, C2->C1, C3->C1, C1->C3)
            MotifType.INCOMING_1TO3_TRIADS.name: {0: MotifType.DYADIC_TRIADS.name,
                                                  1: MotifType.UNIDIRECTIONAL_TRIADS.name,
                                                  2: MotifType.INCOMING_TRIADS.name},

            # (C1, C2, C3, C1->C2, C1->C3, C3->C1)
            MotifType.OUTGOING_3TO1_TRIADS.name: {0: MotifType.DYADIC_TRIADS.name,
                                                  1: MotifType.UNIDIRECTIONAL_TRIADS.name,
                                                  2: MotifType.OUTGOING_TRIADS.name},

            # (C1, C2, C3, C2->C1, C3->C1, C2->C3, C3->C2)
            MotifType.INCOMING_RECIPROCAL_TRIADS.name: {0: MotifType.OUTGOING_3TO1_TRIADS.name,
                                                        1: MotifType.OUTGOING_3TO1_TRIADS.name,
                                                        2: MotifType.INCOMING_2TO3_TRIADS.name,
                                                        3: MotifType.INCOMING_2TO3_TRIADS.name},

            # (C1, C2, C3, C1->C2, C1->C3, C2->C3, C3->C2)
            MotifType.OUTGOING_RECIPROCAL_TRIADS.name: {0: MotifType.INCOMING_1TO3_TRIADS.name,
                                                        1: MotifType.INCOMING_1TO3_TRIADS.name,
                                                        2: MotifType.INCOMING_2TO3_TRIADS.name,
                                                        3: MotifType.INCOMING_2TO3_TRIADS.name},

            # (C1, C2, C3, C1->C2, C2->C3, C3->C1, C1->C3)
            MotifType.DIRECTED_CYCLE_1TO3_TRIADS.name: {0: MotifType.INCOMING_1TO3_TRIADS.name,
                                                        1: MotifType.OUTGOING_3TO1_TRIADS.name,
                                                        2: MotifType.INCOMING_2TO3_TRIADS.name,
                                                        3: MotifType.DIRECTED_CYCLE_TRIADS.name},

            # (C1, C2, C3, C1->C2, C2->C1, C1->C3, C3->C1)
            MotifType.DIRECIPROCAL_TRIADS.name: {0: MotifType.INCOMING_1TO3_TRIADS.name,
                                                 1: MotifType.OUTGOING_3TO1_TRIADS.name,
                                                 2: MotifType.INCOMING_1TO3_TRIADS.name,
                                                 3: MotifType.OUTGOING_3TO1_TRIADS.name},

            # (C1, C2, C3, C1->C2, C1->C3, C2->C1, C3->C1, C2->C3)
            MotifType.DIRECIPROCAL_2TO3_TRIADS.name: {0: MotifType.OUTGOING_RECIPROCAL_TRIADS.name,
                                                      1: MotifType.DIRECTED_CYCLE_1TO3_TRIADS.name,
                                                      2: MotifType.DIRECTED_CYCLE_1TO3_TRIADS.name,
                                                      3: MotifType.INCOMING_RECIPROCAL_TRIADS.name,
                                                      4: MotifType.DIRECIPROCAL_TRIADS.name},

            # (C1, C2, C3, C1->C2, C2->C1, C2->C3, C3->C2, C3->C1, C1->C3)
            MotifType.TRIRECIPROCAL_TRIADS.name: {0: MotifType.DIRECIPROCAL_2TO3_TRIADS.name,
                                                  1: MotifType.DIRECIPROCAL_2TO3_TRIADS.name,
                                                  2: MotifType.DIRECIPROCAL_2TO3_TRIADS.name,
                                                  3: MotifType.DIRECIPROCAL_2TO3_TRIADS.name,
                                                  4: MotifType.DIRECIPROCAL_2TO3_TRIADS.name,
                                                  5: MotifType.DIRECIPROCAL_2TO3_TRIADS.name}
        }

    @staticmethod
    def relations():
        """
        :return: dictionary where key is parent motif name, and value is a list of child motif names
        """
        return {
            MotifType.NO_EDGE_TRIADS.name: [MotifType.SINGLE_EDGE_TRIADS.name],
            MotifType.SINGLE_EDGE_TRIADS.name: [MotifType.UNIDIRECTIONAL_TRIADS.name,
                                                MotifType.DYADIC_TRIADS.name,
                                                MotifType.OUTGOING_TRIADS.name,
                                                MotifType.INCOMING_TRIADS.name],
            MotifType.UNIDIRECTIONAL_TRIADS.name: [MotifType.OUTGOING_3TO1_TRIADS.name,
                                                   MotifType.DIRECTED_CYCLE_TRIADS.name,
                                                   MotifType.INCOMING_1TO3_TRIADS.name,
                                                   MotifType.INCOMING_2TO3_TRIADS.name],
            MotifType.DYADIC_TRIADS.name: [MotifType.OUTGOING_3TO1_TRIADS.name,
                                           MotifType.INCOMING_1TO3_TRIADS.name],
            MotifType.OUTGOING_TRIADS.name: [MotifType.OUTGOING_3TO1_TRIADS.name,
                                             MotifType.INCOMING_2TO3_TRIADS.name],
            MotifType.INCOMING_TRIADS.name: [MotifType.INCOMING_1TO3_TRIADS.name,
                                             MotifType.INCOMING_2TO3_TRIADS.name],
            MotifType.OUTGOING_3TO1_TRIADS.name: [MotifType.DIRECIPROCAL_TRIADS.name,
                                                  MotifType.DIRECTED_CYCLE_1TO3_TRIADS.name,
                                                  MotifType.INCOMING_RECIPROCAL_TRIADS.name],
            MotifType.DIRECTED_CYCLE_TRIADS.name: [MotifType.DIRECTED_CYCLE_1TO3_TRIADS.name],
            MotifType.INCOMING_1TO3_TRIADS.name: [MotifType.DIRECIPROCAL_TRIADS.name,
                                                  MotifType.DIRECTED_CYCLE_1TO3_TRIADS.name,
                                                  MotifType.OUTGOING_RECIPROCAL_TRIADS.name],
            MotifType.INCOMING_2TO3_TRIADS.name: [MotifType.DIRECTED_CYCLE_1TO3_TRIADS.name,
                                                  MotifType.OUTGOING_RECIPROCAL_TRIADS.name,
                                                  MotifType.INCOMING_RECIPROCAL_TRIADS.name],
            MotifType.DIRECIPROCAL_TRIADS.name: [MotifType.DIRECIPROCAL_2TO3_TRIADS.name],
            MotifType.DIRECTED_CYCLE_1TO3_TRIADS.name: [MotifType.DIRECIPROCAL_2TO3_TRIADS.name],
            MotifType.OUTGOING_RECIPROCAL_TRIADS.name: [MotifType.DIRECIPROCAL_2TO3_TRIADS.name],
            MotifType.INCOMING_RECIPROCAL_TRIADS.name: [MotifType.DIRECIPROCAL_2TO3_TRIADS.name],
            MotifType.DIRECIPROCAL_2TO3_TRIADS.name: [MotifType.TRIRECIPROCAL_TRIADS.name],
            MotifType.TRIRECIPROCAL_TRIADS.name: []
        }

    @staticmethod
    def transitions():
        """
        :return: Dictionary of Key: Motif transition (MotifType.name, MotifType.name), Value: # of transitions (Int)
                with all values set to 0
        """
        retval = dict()

        for parent, children in TriadMotif.relations().items():
            retval[(parent, parent)] = 0
            for c in children:
                retval[(parent, c)] = 0

        return retval

    def visualize(self, text_limit: Optional[int] = 20, verbose: bool = False) -> None:
        """
        Uses Graphviz to construct a visualisation of the motif, with edges labelled with the utterance text
        :param text_limit:
        :param verbose:
        :return:
        """
        g = Digraph('G')
        g.attr(rankdir='LR')
        edges = sorted([e[0] for e in self.edges], key=lambda x: x['timestamp'])
        for idx, edge in enumerate(edges):
            if verbose:
                label = "{}. {}".format(idx + 1, edge['text']) if text_limit is None else "{}. {}".format(idx + 1, edge['text'][:text_limit])
            else:
                label = str(idx+1)
            g.edge(str(edge['speaker']), str(edge['target']), label=label)
        g.view()

    def get_development_path(self) -> Tuple[str]:
        """
        :return: Returns a Tuple of the different stages of the motif's development
        """
        path = []
        curr_motif_state = self
        while curr_motif_state is not None:
            path.append(curr_motif_state.triad_type)
            curr_motif_state = curr_motif_state.regress()
        return tuple(path[::-1])

    def replay_motif(self) -> None:
        """
        Prints each utterance in order and the corresponding motif development
        :return: None
        """
        pathway = self.get_development_path()
        sorted_edges = sorted(self.edges, key=lambda e: e[0]['timestamp'])
        for i in range(len(pathway)):
            if i == 0: continue
            print("########################")
            print("{} -> {}".format(pathway[i-1], pathway[i]))
            print()
            edge = sorted_edges[i-1][0]
            print("{} -> {}".format(edge['speaker'], edge['target']))
            print()
            print("TEXT: {}".format(edge['text']))
            print()
            if edge['root']:
                print("(This utterance responds to a top-level-comment!)")
                print()
