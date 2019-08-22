from functools import total_ordering
from typing import Dict, List, Collection, Hashable, Callable, Set, Generator, Tuple, Optional, ValuesView

@total_ordering
class User:
    """Represents a single user in a dataset.

    :param name: name of the user.
    :type name: str
    :param utts: dictionary of utterances by the user, where key is user id
    :param convos: dictionary of conversations started by the user, where key is conversation id
    :param meta: arbitrary dictionary of attributes associated
        with the user.
    :type meta: dict

    :ivar name: name of the user.
    :ivar meta: dictionary of attributes associated with the user.
    """

    def __init__(self, name: str=None, utts=None, convos=None, meta: Optional[Dict]=None):
        self._name = name
        self.utterances = utts if utts is not None else dict()
        self.conversations = convos if convos is not None else dict()
        self._meta = meta if meta is not None else {}
        self._split_attribs = set()
        self._update_uid()

    def identify_by_attribs(self, attribs: Collection) -> None:
        """Identify a user by a list of attributes. Sets which user info
        attributes should distinguish users of the same name in equality tests.
        For example, in the Supreme Court dataset, users are labeled with the
        current case id. Call this method with attribs = ["case"] to count
        the same person across different cases as different users.

        By default, if this function is not called, Users are identified by name only.

        :param attribs: Collection of attribute names.
        :type attribs: Collection
        """

        self._split_attribs = set(attribs)
        self._update_uid()

    def _get_name(self): return self._name

    def _set_name(self, value: str):
        self._name = value
        self._update_uid()

    name = property(_get_name, _set_name)

    def get_utterance_ids(self) -> List[Hashable]:
        return list(self.utterances.keys())

    def get_utterance(self, ut_id: Hashable): #-> Utterance:
        return self.utterances[ut_id]

    def iter_utterances(self): #-> Generator[Utterance, None, None]:
        for v in self.utterances.values():
            yield v

    def get_conversation_ids(self) -> List[str]:
        return list(self.conversations.keys())

    def get_conversation(self, cid: Hashable): # -> Conversation:
        return self.conversations[cid]

    def iter_conversations(self): # -> Generator[Conversation, None, None]:
        for v in self.conversations.values():
            yield v

    def _get_meta(self): return self._meta

    def _set_meta(self, value: Dict):
        self._meta = value
        self._update_uid()
    meta = property(_get_meta, _set_meta)

    def add_meta(self, key: Hashable, value) -> None:
        self.meta[key] = value

    def _update_uid(self):
        rep = dict()
        rep["name"] = self._name
        if self._split_attribs:
            rep["attribs"] = {k: self._meta[k] for k in self._split_attribs
                              if k in self._meta}
        self._uid = "User(" + str(sorted(rep.items())) + ")"

    def __eq__(self, other):
        return self._uid == other._uid

    def __lt__(self, other):
        return self._uid < other._uid

    def __hash__(self):
        return hash(self._uid)

    def __repr__(self):
        return self._uid

    # def copy(self):
    #     """
    #     :return: A duplicate of the User with the same data and metadata
    #     """
    #     return User(name=self.name, utts=self.utterances, convos=self.conversations, meta=self.meta.copy())
