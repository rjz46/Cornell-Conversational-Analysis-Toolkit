import convokit
print(convokit)
import numpy as np

# create corpus object
corpus = convokit.Corpus(filename=convokit.download("reddit-corpus-small"))

# we typically would not need to expose make_hypergraph publicly, but we do this here
# to demonstrate Hypergraph methods

G = convokit.HyperConvo()._make_hypergraph(corpus)

def summarize_dist(name, l):
    print("{}: min {}, mean {:.4f}, max {}".format(
        name, min(l), np.mean(l), max(l)))

# in- and outdegree distributions
summarize_dist("user to user indegrees", G.indegrees(True, True))
summarize_dist("user to user outdegrees", G.outdegrees(True, True))
summarize_dist("user to comment indegrees", G.indegrees(True, False))
summarize_dist("user to comment outdegrees", G.outdegrees(True, False))
summarize_dist("comment to comment indegrees", G.indegrees(False, False))
summarize_dist("comment to comment outdegrees", G.outdegrees(False, False))
print()

def summarize_motifs(name, l):
    print("{}: count {}".format(name, len(l)))

# motif extraction
summarize_motifs("reciprocity", G.reciprocity_motifs())
summarize_motifs("external reciprocity", G.external_reciprocity_motifs())
summarize_motifs("dyadic interaction", G.dyadic_interaction_motifs())
summarize_motifs("incoming triad", G.incoming_triad_motifs())
summarize_motifs("outgoing triad", G.outgoing_triad_motifs())
print()

print("example reciprocity motif:", G.reciprocity_motifs()[0])
print()

hc = convokit.HyperConvo()
# HyperConvo interface: get high-level degree features
hc.fit_transform(corpus)

feats = dict()
convos = corpus.iter_conversations()

for convo in convos:
    feats.update(convo.meta['hyperconvo'])

random_thread = next(iter(feats))

for k, v in feats[random_thread].items():
    print("{}: {:.4f}".format(k, v))
print()
