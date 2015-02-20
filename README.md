# dm-graphs

The scripts in `dm_graphs.py` provide a way to explore the relationships between words in a distributional model. It relies on the `NetworkX` package. Please see the paper for more context.

## Basic usage
The file `example.py` shows how to use the `dm_graphs` module. It initializes a generator (based on the model in `toy_model.py`) that produces tuples `(u,v,w)` corresponding to edges between `u` and `v` with weight `w`. The weight in this case is the cosine distance between `u` and `v`.

After initializing the generator, we can create a network containing the edges it yields. The rest of the code shows how to make the network easier to visualize by using `dm_graphs.graph_reduce(G)` and `dm_graphs.MST_pathfinder(G)`. These return a sparser version of the network.

## Features

**Main functions**
* `MST_pathfinder()` is an implementation of MST-pathfinder algorithm (Quirin et al. 2008).
* `graph_reduce()` reduces the graph by only including the edges that link each node to its top-n similar neighbors. It also has an optional restriction such that every edge should have a weight above a particular threshold.

**Graph analysis**
* `main_graph()` returns the largest connected component.
* `graph_analysis()` returns some statistics about the graph, including a suggested partition based on the Louvain method.

**Utilities**
* `add_partition_data()` adds partition data from the analysis to the graph.
* `invert_weights()` changes weights on all edges to 1-weight.
* `rank_reweight()` reweights the edges based on the similarity ranks of the nodes.
* `remove_weights()` removes the weights from the graph. Useful if you don't want Gephi to make the edges thicker.
* `write_functions()` displays the available methods to write out the graphs.
