import networkx as nx
import community
from itertools import combinations

################################################################################
#   Utilities

def invert_weights(G):
    """Invert weights. This is useful when weights are Cosine DISTANCE rather than
    Cosine SIMILARITY. Be sure to check whether the weights positively correspond
    to connection strength, or else Gephi will generate awkward pictures."""
    NG = nx.Graph()
    NG.add_weighted_edges_from((a,b,1-G[a][b]['weight']) for a,b in G.edges())
    return NG

def remove_weights(G):
    """Remove weights. This is useful if you want to write out the graph without
    any weighting information. For example, if you don't want Gephi to make edges
    thicker."""
    NG = nx.Graph()
    NG.add_edges_from(G.edges())
    return NG

def add_partition_data(G,partition,name="Part"):
    """Add partition data to the graph. The partition should come in the form of
    a dictionary, with k=node, v=part. Optionally, you can specify a name for the
    partitioning."""
    NG = G.copy()
    for node in NG.nodes():
        NG.node[node]['Part'] = partition[node]
    return NG

def rank_dict(G,node):
    "Returns a dictionary of words connected to the node with their ranking."
    ranked          = dict()
    neighbors       = G.neighbors(node)
    num_neighbors   = float(len(neighbors))
    for rank,word in enumerate( sorted(neighbors,
                                key=lambda x:G[node][x]['weight'])):
        ranked[word] = (rank+1)/num_neighbors
    return ranked

def rank_reweight(G):
    "Reweight the graph by rank rather than similarity."
    NG  = nx.Graph()
    d   = {node:rank_dict(G,node) for node in G.nodes()}
    def avg_rank(a,b,d):
        return (d[a][b] + d[b][a])/2
    NG.add_weighted_edges_from((a,b,avg_rank(a,b,d)) for a,b in G.edges())
    return NG

def write_functions():
    "See what functions you can use to write out the graph."
    print [f for f in dir(nx) if f.startswith('write')]
    
################################################################################
#   Analyzing the graph

def main_graph(G):
    "Takes a graph object and returns the largest graph."
    graph_list = list(nx.connected_component_subgraphs(G))
    return sorted(graph_list,   # Main graph is equal to the first element
                  reverse=True, # of the list of isolated graphs, sorted in
                  key=len       # reverse order.
                        )[0]

def graph_analysis(G):
    """Analyze graph. Returns a dictionary with useful data.
    Cannot deal with weights below 0, so all negative weights are set to 0."""
    MG                      = main_graph(G)
    for a,b in MG.edges():
        w = MG[a][b]['weight']
        if w < 0:
            MG[a][b]['weight'] = 0
    partition = community.best_partition(MG)
    return { 'num_clusters':    max(partition.values()),
             'modularity':      community.modularity(partition,MG),
             'size':            len(MG.nodes()),
             'partition':       partition}

def partition_by_group(partition):
    "Get a dictionary with k=partition, v=[words,in,partition]"
    d = {i:[] for i in range(max(partition.values())+1)}
    for w in partition:
        d[partition[w]].append(w)
    return d

################################################################################
#   Reducing the graph

def graph_reduce(G,n=5,theta=0.5,
                    top_n=True,
                    threshold=False,
                    return_only_main=True):
    """Reduce the graph through either (or both) of two similarity measures:
        - An edge can only connect A and B iff B is in A's top-n similar nodes,
        or vice versa.
        - An edge can only connect A and B iff the weight is sufficient."""
    
    if top_n:                             # if we are using the top-n measure
        NG = nx.Graph()                   # generate a new graph
        for node in G:                    # and for each node a sorted list of
            l = sorted( G[node].keys(),   # similar nodes.
                    key=lambda x:G[node][x]['weight'],
                    reverse=True)[:n]     # first n items
            
            # And add the top-n similar edges to the graph!
            NG.add_weighted_edges_from((node,x,G[node][x]['weight']) for x in l)
    else:   # Else just create a copy of G, that we can prune using the threshold
            # measure.
        NG = G.copy()
    
    # Code for the threshold measure:
    if threshold:                         # if threshold is active, and weights
        for a,b in NG.edges():            # for all connected nodes a,b
            if G[a][b]['weight'] < theta: # if the similarity is too small.
                NG.remove_edge(a,b)       # remove the edge between a and b
    
    # By default we only return the main graph and ignore isolated graphs:
    if return_only_main:
        return main_graph(NG)
    else:
        return NG


def MST_pathfinder(G):
    """The MST-pathfinder algorithm (Quirin et al. 2008) reduces the graph to the
    unions of all minimal spanning trees."""
    NG    = nx.Graph()
    edges = sorted( ((G[a][b]['weight'],a,b) for a,b in G.edges()),
                        reverse=True) # reversed, because greater similarity = better
    clusters = {node:i for i,node in enumerate(G.nodes())}
    while not edges == []:
        w1,a,b = edges[0]
        l      = []
        # Select edges to be considered this round:
        for w2,u,v in edges:
            if w1 == w2:
                l.append((u,v,w2))
            else:
                break
        # Remaining edges are those not being considered this round:
        edges = edges[len(l):]
        # Only select those edges for which the items are not in the same cluster
        l = [(a,b,c) for a,b,c in l if not clusters[a]==clusters[b]]
        # Add these edges to the graph:
        NG.add_weighted_edges_from(l)
        # Merge the clusters:
        for a,b,w in l:
            cluster_1 = clusters[a]
            cluster_2 = clusters[b]
            clusters = {node:cluster_1 if i==cluster_2 else i
                        for node,i in clusters.iteritems()}
    return NG
