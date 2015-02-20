from __future__ import print_function
import networkx as nx   # NetworkX provides the graph functionality we need.
import dm_graphs        # Main module.
import toy_model        # Provides toy data to work with.

# Get the data. In this case, we use a model based on tags from Freesound.org
# The generator produces tuples: (node_1, node_2, weight)
gen = toy_model.get_toy_data()

# Define a graph object.
G   = nx.Graph()

# Fill the graph with the data.
G.add_weighted_edges_from(gen)

# But! Weights correspond to distance rather than similarity!
# We invert the graph using the invert_weights function.
# Note that this function just takes 1-weight as the new weight.
G = dm_graphs.invert_weights(G)

# Get the main connected component.
MG  = dm_graphs.main_graph(G)

# Reduce the network using Top-N reduction (default, n=5).
TOPN = dm_graphs.graph_reduce(MG)

# Reduce the network using the MST-pathfinder algorithm.
# This takes quite a long time for larger graphs, but is very fast for small ones.
# So we reduce the top-N network further.
MST  = dm_graphs.MST_pathfinder(TOPN)

# Writing the graph to a file:
# ------------------------------------------------------------------------------
nx.write_gexf(MST,'toy_graph.gexf')
