"""
Generates weighted or unweighted graphs using a block model.
"""

import networkx as nx 
import numpy as np 
import scipy.stats as stats
import sys
import random as rnd

def unweighted_two_community_graph(N, mu, avg_degree):
    """
    To create the graph we generate a poisson distributed number of edges for each 
    group pair where the mean is w_rs  (1/2 w_rs when r=s). Then we assign each end
    of the vertex to the proper group randomly. 
    w_rs = p_rs * (# members in group s * # members in group r)
    """

    # Calculate block matrix components (assumes communities of same size)
    B = 2.0 * avg_degree * mu / N # Off diagonal probability
    A = 2.0 * avg_degree / N - B # On diagonal probability

    # Random group member assignment
    nodes = set(xrange(N))
    community1 = set(rnd.sample(nodes, N/2))
    community2 = nodes.difference(community1)
    graph = nx.DiGraph()
    graph.add_nodes_from(nodes)
    for node in graph.nodes_iter():
        if node in community1:
            graph.node[node]['community'] = 1
        else:
            graph.node[node]['community'] = 2

    # Generate edges
    num_edges_com1tocom2, num_edges_com2tocom1 = np.random.poisson(B * (N/2.0)**2, size=2)
    num_edges_com1tocom1, num_edges_com2tocom2 = np.random.poisson(A * (N/2.0)**2, size=2)

    add_edges_to_graph(graph, community1, community2, num_edges_com1tocom2)
    add_edges_to_graph(graph, community2, community1, num_edges_com2tocom1)
    add_edges_to_graph(graph, community1, community1, num_edges_com1tocom1)
    add_edges_to_graph(graph, community2, community2, num_edges_com2tocom2)

    return graph

def add_edges_to_graph(graph, com1_set, com2_set, num_edges_to_add):
    com1_set = list(com1_set)
    com2_set = list(com2_set)
    for i in xrange(num_edges_to_add):
        tail = rnd.choice(com1_set)
        head = rnd.choice(com2_set)
        tries = 0
        while ( (graph.has_edge(tail, head) or (tail==head)) and (tries < 10)):
            tail = rnd.choice(com1_set)
            head = rnd.choice(com2_set)
            tries += 1

        if tries >= 10:
            continue
        else:
            graph.add_edge(tail, head)

def uniform_weighted_two_community_graph(N, mu, avg_degree, lower_bound=0, upper_bound=1):

    graph = unweighted_two_community_graph(N, mu, avg_degree)
    weights = np.random.uniform(lower_bound, upper_bound, size=nx.number_of_edges(graph))
    for i, edge in enumerate(graph.edges_iter()):
        graph[edge[0]][edge[1]]['weight'] = weights[i]

    return graph

def gamma_weighted_two_community_graph(N, mu, avg_degree, EE_W=30000.0, negative_weights=False):

    graph = unweighted_two_community_graph(N, mu, avg_degree)

    weights = random_gamma(EE_W, nx.number_of_edges(graph))

    for i, edge in enumerate(graph.edges_iter()):
        if rnd.random() < 0.5:
            graph[edge[0]][edge[1]]['weight'] = weights[i]
        else:
            graph[edge[0]][edge[1]]['weight'] = -weights[i]

    return graph

def random_gamma(mean, size=1, shape=4.0):
    """
    Determines the appropriate gamma parameters for a given mean and 
    standard deviation, then returns an array of randomly generated
    values following that distribution.

    For std = mean / 2
    """

    if mean > 0:
        shape = shape
        scale = mean / shape
        return stats.gamma.rvs(shape, size=size, loc=0.0, scale=scale)
    else:
        mean = -1. * mean
        shape = shape
        scale = mean / shape
        return -1. * stats.gamma.rvs(shape, size=size, loc=0.0, scale=scale)

if __name__ == '__main__':
    
    import time
    start_time = time.time()
    graph = unweighted_two_community_graph(250, 0.02, 7)
    print("--- %s seconds ---" % (time.time() - start_time))
    nx.write_gexf(graph, "test.gexf")