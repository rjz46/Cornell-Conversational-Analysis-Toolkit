from typing import Dict, List, Tuple, Set
from convokit.model import Utterance, User
import random

"""
Given a thread of form Dict[utterance_id (str) -> Utterance],
create a thread using the same arrival pattern with random reply_tos to any antecedent participant, but no self-replies
"""


def get_arrival_order(thread: Dict[str, Utterance]) -> List[User]:
    """
    Gets arrival order of Users from thread
    :param thread: utterances of a thread
    :return: ordered list of Users according to when they joined the thread (earliest -> last)
    """
    retval = []
    for utt_id, utt in thread.items():
        retval.append((utt.get('user'), utt.get('timestamp')))
    return [user for user, ts in sorted(retval, key=lambda x: x[1])]


def random_other_utt(prev_utts: List[Utterance], responses: Dict[str, Set[User]], responder: User) -> Utterance:
    """
    Randomly selects one of the previous utterances for the User to respond to + *that the User has not already responded to)
    :param prev_utts: List of previous utterances
    :param responses: Dictionary tracking which Users have responded to each utterance
    :param responder: User looking for an utterance (that is by some other User and that User has not already responded to) to respond to
    :return: one of the previous utterances (for the responder to respond to). root utterance if no such utterances meet the conditions
    """
    # print("Looking for utt for {} to respond to".format(responder))
    other_utts = list(filter(lambda x: x.get("user") != responder, prev_utts)) # avoid responding to self if possible
    # print("Filter responding to self: {}".format(other_utts))

    other_utts = list(filter(lambda x: responder not in responses[x.id], other_utts)) # avoid responding to same Utterance more than once
    # print("Filter responding twice: {}".format(other_utts))
    if len(other_utts) == 0: return prev_utts[0] # reply to root/self if no other

    return random.choice(other_utts)

def randomize_thread(root: str, thread: Dict[str, Utterance]) -> Dict[str, Utterance]:
    order = get_arrival_order(thread)

    # First utterance in thread is always the root
    utt0 = Utterance(id=root, user=order[0], root=root, reply_to=None, timestamp=0)

    prev_utts = [utt0]
    responses = dict()
    responses[utt0.id] = set()

    for idx, user in enumerate(order[1:]):
        utt_to_respond_to = random_other_utt(prev_utts, responses, user)
        responses[utt_to_respond_to.id].add(user)

        new_utt = Utterance(id="{}_{}".format(root, idx), user=user, root=root, reply_to=utt_to_respond_to.get("id"), timestamp=idx)
        prev_utts.append(new_utt)
        responses[new_utt.id] = set()

    return {utt.get("id"): utt for utt in prev_utts}