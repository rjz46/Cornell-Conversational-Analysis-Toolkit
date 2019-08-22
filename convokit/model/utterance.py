from typing import Dict, List, Collection, Hashable, Callable, Set, Generator, Tuple, Optional, ValuesView
from .user import User

class Utterance:
    """Represents a single utterance in the dataset.

    :param id: the unique id of the utterance. Can be any hashable type.
    :param user: the user giving the utterance.
    :param root: the id of the root utterance of the conversation.
    :param reply_to: id of the utterance this was a reply to.
    :param timestamp: timestamp of the utterance. Can be any
        comparable type.
    :param text: text of the utterance.
    :type text: str

    :ivar id: the unique id of the utterance.
    :ivar user: the user giving the utterance.
    :ivar root: the id of the root utterance of the conversation.
    :ivar reply_to: id of the utterance this was a reply to.
    :ivar timestamp: timestamp of the utterance.
    :ivar text: text of the utterance.
    """

    def __init__(self, id: Optional[Hashable]=None, user: Optional[User]=None,
                 root: Optional[Hashable]=None, reply_to: Optional[Hashable]=None,
                 timestamp: Optional[int]=None, text: Optional[str]=None,
                 meta: Optional[Dict]=None):
        self.id = id
        self.user = user
        self.root = root
        self.reply_to = reply_to
        self.timestamp = timestamp
        self.text = text
        self.meta = meta if meta is not None else {}

    def get(self, key: str):
        if key == "id":
            return self.id
        elif key == "user":
            return self.user
        elif key == "root":
            return self.root
        elif key == "reply_to":
            return self.reply_to
        elif key == "timestamp":
            return self.timestamp
        elif key == "text":
            return self.text
        elif key == "meta":
            return self.meta

    # def copy(self):
    #     """
    #     :return: A duplicate of this Utterance with the same data and metadata
    #     """
    #     return Utterance(id=self.id,
    #                      user=self.user,
    #                      root=self.root,
    #                      reply_to=self.reply_to,
    #                      timestamp=self.timestamp,
    #                      text=self.text,
    #                      meta=self.meta.copy())

    def add_meta(self, key: Hashable, value) -> None:
        self.meta[key] = value

    def __eq__(self, other):
        return self.__dict__ == other.__dict__

    def __repr__(self):
        return "Utterance(" + str(self.__dict__) + ")"
