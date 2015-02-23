import networkx as nx
import dm_graphs

from itertools import combinations
from gensim.models import Word2Vec
from nltk.corpus import wordnet as wn

# Load the Google news vectors, download from https://code.google.com/p/word2vec/
googlenews = '/path/to/GoogleNews-vectors-negative300.bin.gz'
model   = Word2Vec.load_word2vec_format(googlenews, binary=True)

# Get animal vocabulary, which is the intersection between the vocabulary and
# all hyponyms of 'animal.n.01' in WordNet:
vocab   = set(model.vocab.keys())
animal  = wn.synset('animal.n.01')
animals = set([w for s in animal.closure(lambda s:s.hyponyms())
                 for w in s.lemma_names])
overlap = animals & vocab

# Create edge generator:
edges   = ((a,b,model.similarity(a,b)) for a,b in combinations(overlap,2))

# Create graph and add edges from the generator:
G       = nx.Graph()
G.add_weighted_edges_from(edges)

# Reduce the graph:
NG      = dm_graphs.graph_reduce(G)
MST     = dm_graphs.MST_pathfinder(NG)

# Write out the results:
nx.write_gexf(NG,"googlenews.gexf")
nx.write_gexf(MST,"googlenews_MST.gexf")
