import inspect
import unweighted_directed_lfr_benchmark as lfr
import networkx as nx
import numpy as np

def uniform_weighted_directed_lfr_graph(N, mu, k, maxk, minc, maxc, weight_bounds, **kwargs):
    """
    """

    multicommunityLFR_pardict = { key: value for key, value in kwargs.items() \
        if key in inspect.getargspec(lfr.unweighted_directed_lfr_graph).args }

    reservoir = lfr.unweighted_directed_lfr_graph(N, mu, k, maxk, minc, maxc, **multicommunityLFR_pardict)
    weights = np.random.uniform(weight_bounds[0], weight_bounds[1], size=nx.number_of_edges(reservoir))
    for i, edge in enumerate(reservoir.edges_iter()):
        reservoir[edge[0]][edge[1]]['weight'] = weights[i]

    return reservoir

def uniform_weighted_directed_lfr_graph_asarray(**kwargs):
    """
    """

    return np.asarray(nx.to_numpy_matrix(make_reservoir(**kwargs)))


def uniform_weighted_directed_lfr_graph_asnx(**kwargs):
    """
    """

    return make_reservoir(**kwargs)

if __name__ == '__main__':
    
    print(uniform_weighted_directed_lfr_graph_asarray(N=50, mu=0.1, k=4, maxk=4, minc=25, maxc=25, weight_bounds=(0.5, 1.0)))