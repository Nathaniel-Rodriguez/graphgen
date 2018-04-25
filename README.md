Contains a module for generating hierarchical modular networks, a stochastic block model, a two community block model, and LFR benchmark graphs. Install using either setuptools or pip:

```bash
pip install git+https://github.com/Nathaniel-Rodriguez/graphgen.git
``` 

or with setuptools after downloading and unpacking:

```bash
python setup.py install
```

The LFR graphs use code from Andrea Lancichinetti that has been lightly modified, replacing the file based input/output with a Python API.

The original code for the LFR graphs can be found here:
	https://sites.google.com/site/andrealancichinetti/files

For LFR graphs, there are all combinations of weighted/unweighted and directed/undirected available. All are in the `lfr_generators` module.

For example, to make a weighted, directed network one can call:

```python
from graphgen.lfr_generators import weighted_directed_lfr_graph

edge_array, community_memberships, weights = weighted_directed_lfr_graph(
                                                num_nodes=100, 
                                                average_k=5, 
                                                max_degree=10, 
                                                mut=0.3, 
                                                muw=0.1, 
                                                com_size_min=10, 
                                                com_size_max=30, 
                                                seed=3287439)
```

The `edge_array` is an Ex2 numpy array. `community_memberships` is a tuple with a tuple of community memberships for each node. The ith element in the tuple corresponds to the ith node. All nodes are ID'd from 0 to num_nodes-1. More convenient forms of output are available. To output the network as a networkx graph do:

```python
from graphgen.lfr_generators import weighted_directed_lfr_as_nx

nx_graph = weighted_directed_lfr_as_nx(num_nodes=100, 
                                       average_k=5, 
                                       max_degree=10, 
                                       mut=0.3, 
                                       muw=0.1, 
                                       com_size_min=10, 
                                       com_size_max=30, 
                                       seed=3287439)                                       
```

The weights and node memberships are automatically assigned as attributes to the edges and nodes of the graph. You can use `help(lfr_generators)` to get information about what functions are available.

Currently four base versions of the LFR benchmark graphs are available:

- weighted_undirected_lfr_graph
- weighted_directed_lfr_graph
- unweighted_undirected_lfr_graph
- unweighted_directed_lfr_graph

Use `help()` on these function's doc strings to get argument information. Seeds are required for all of them and are not optional.