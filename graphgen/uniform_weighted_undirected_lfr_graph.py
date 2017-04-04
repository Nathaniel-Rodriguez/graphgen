import inspect
from .unweighted_undirected_lfr_benchmark import unweighted_undirected_lfr_graph
import networkx as nx
import numpy as np

def uniform_weighted_undirected_lfr_graph(N, mu, k, maxk, minc, maxc, 
                                                    weight_bounds, **kwargs):
    """
    """
    
    pardict = { key: value 
        for key, value in kwargs.items()
        if key in inspect.getargspec(
                                unweighted_undirected_lfr_graph).args }

    reservoir = unweighted_undirected_lfr_graph(
                                        N, mu, k, maxk, minc, maxc, **pardict)
    weights = np.random.uniform(weight_bounds[0], 
                        weight_bounds[1], size=nx.number_of_edges(reservoir))
    for i, edge in enumerate(reservoir.edges_iter()):
        reservoir[edge[0]][edge[1]]['weight'] = weights[i]

    return reservoir

def uniform_weighted_undirected_lfr_graph_asarray(**kwargs):
    """
    """

    return np.asarray(
            nx.to_numpy_matrix(uniform_weighted_undirected_lfr_graph(**kwargs)))


def uniform_weighted_undirected_lfr_graph_graph_asnx(**kwargs):
    """
    """

    return uniform_weighted_undirected_lfr_graph(**kwargs)

if __name__ == '__main__':
    
    print(unweighted_undirected_lfr_graph_asarray(N=50, mu=0.1, 
                                                    k=4, maxk=4, minc=25))