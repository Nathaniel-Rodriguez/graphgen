"""
Graphs generated follow the i->j connection scheme
"""

import numpy as np 
import networkx as nx 
import scipy.stats as stats
from itertools import product

def generate_discrete_distribution(N, distribution_type, distribution_args=()):
    """
    Generates a sequence of distrete random variables for the 
    degree distribution given the parameters for the distribution
    and the type that is used.

    Currently supports:

        poisson: lambda
        uniform: low, high
        powerlaw: rank, exponent

    """

    if distribution_type == 'poisson':
        return stats.poisson(*distribution_args).rvs(size=N)

    elif distribution_type == 'uniform':
        return stats.randint(*distribution_args).rvs(size=N)

    elif distribution_type == 'powerlaw' or distribution_type == 'zipf':
        return stats.zipf(*distribution_args).rvs(size=N)

def generate_continuous_distribution(E, distribution_type, distribution_args=(), 
    sign_flip_fraction=0.0):
    """
    Generates a sequence of continuous weights for each edge.
    It also supports a fraction that are made negative.
    This is accomplished by randomly flipping the sign of a 
    fraction of the drawn values. Depending upon the
    distribution chosen, negative weights maybe generated
    anyway (e.g. a normal distribution).

    Currently supports:

        normal: beta, (loc,scale)
        uniform: low, scale, (loc,scale)
        pareto: b, (loc,scale)
        gamma: a, (loc,scale)
        lognormal: s, (loc,scale)

    """

    if distribution_type == "norm" or distribution_type == "normal":
        rvs = stats.gennorm(*distribution_args).rvs(size=E)

    elif distribution_type == "uniform":
        rvs = stats.uniform(*distribution_args).rvs(size=E)

    elif distribution_type == "pareto":
        rvs = stats.pareto(*distribution_args).rvs(size=E)

    elif distribution_type == "gamma":
        rvs = stats.gamma(*distribution_args).rvs(size=E)

    elif distribution_type == "lognorm":
        rvs = stats.lognorm(*distribution_args).rvs(size=E)

    num_to_flip = int(E * sign_flip_fraction)
    indices_to_flip = np.random.choice(range(E), size=num_to_flip, replace=False)
    rvs[indices_to_flip] = -1 * rvs[indices_to_flip]

    return rvs

def calculate_node_connection_probabilities(node_degrees):
    total_group_degree = np.sum(node_degrees)
    return [ expected_degree / float(total_group_degree) 
            for expected_degree in node_degrees ]

def calculate_expected_degrees(group_sizes, 
    degree_distribution_parameter_vector, 
    degree_distribution, correlated_inout_degree):

    num_groups = len(group_sizes)
    expected_node_indegrees_by_group = \
        [ generate_discrete_distribution(group_sizes[group], 
            degree_distribution, degree_distribution_parameter_vector[group]) 
        for group in range(num_groups) ]

    if not correlated_inout_degree:

        expected_node_outdegrees_by_group = \
        [ generate_discrete_distribution(group_sizes[group], 
            degree_distribution, degree_distribution_parameter_vector[group]) 
        for group in range(num_groups) ]
    else:
        expected_node_outdegrees_by_group = expected_node_indegrees_by_group

    return expected_node_indegrees_by_group, expected_node_outdegrees_by_group

def connect_edge_bundle(graph, 
    source_group_size, target_group_size, 
    source_group_nodes, target_group_nodes,
    connection_prob_between_groups,
    source_connection_probs_by_node, target_connection_probs_by_node):

    num_edges = np.random.poisson(connection_prob_between_groups 
                                * source_group_size * target_group_size)
    possible_edges = np.array(list(product(source_group_nodes, 
                                            target_group_nodes)))
    edge_probabilities = [ pair[0] * pair[1] 
        for pair in product(source_connection_probs_by_node, 
                            target_connection_probs_by_node)]
    # Chose only up to # of non-zero
    if num_edges > np.count_nonzero(edge_probabilities):
        num_edges = np.count_nonzero(edge_probabilities)
    chosen_edges = np.random.choice(len(possible_edges), 
                        size=num_edges, replace=False, p=edge_probabilities)

    graph.add_edges_from(possible_edges[chosen_edges])

    return possible_edges[chosen_edges]

def add_connection_weights(graph, edges, distribution_args, 
    distribution_type, flip_fraction):

    if len(edges) != 0:
        weights = generate_continuous_distribution(len(edges), 
                                                distribution_type, 
                                                distribution_args, 
                                                flip_fraction)
        nx.set_edge_attributes(graph, 'weight', { tuple(edges[i]) : weights[i] 
                                                for i in range(len(edges)) })

def add_edge_attributes(graph, edge_bundles_by_group, edge_attribute_dict):

    if edge_attribute_dict['distribution_type'] == 'discrete':
        rvs_generator = self.generate_discrete_distribution
    if edge_attribute_dict['distribution_type'] == 'continuous':
        rvs_generator = self.generate_continuous_distribution

    for bundle, edges in edge_bundles_by_group.items():
        if len(edges) != 0:
            rvs = rvs_generator(len(edges), 
                    edge_attribute_dict['distribution'],
                    edge_attribute_dict['distribution_param_matrix']\
                                        [bundle[0], bundle[1]])
            nx.set_edge_attributes(graph, edge_attribute_dict['key'],
                {tuple(edges[i]) : rvs[i] for i in range(len(edges))})

def remove_self_loops(graph):

    for node in graph.nodes_iter():
        if graph.has_edge(node,node):
            graph.remove_edge(node,node)

def weighted_directed_stochastic_block_model(N, relative_group_sizes,
    connectivity_block_matrix,
    weight_distribution_parameter_matrix,
    degree_distribution_parameter_vector,
    negative_weight_fraction_matrix=None,
    weight_distribution="gamma", degree_distribution="poisson", 
    correlated_inout_degree=True, self_loops=False,
    other_edge_block_attributes=[]):
    """
    N - number of nodes in the graph
    relative_group_sizes - sequence with magnitudes (will be normalized internally)
    connectivity_block_matrix - matrix, elements are connection probabilities 
                                    (i,j element is from group-j to group-i)
    degree_distribution_parameter_vecdegree_distribution="poisson"tor - degree distribution parameters 
                                            (tuple or seq) for each group
    weight_distribution_parameter_matrix - weight distribution parameters 
                                    (tuple of seq) for each edge bundle (i,j)
    weight_distribution - key for distribution
    degree_distribution - key for distribution
    negative_weight_fraction_matrix - vector of [0,1] values 
            specifying the fraction of sign swaps for each edge bundle (i,j)

    Generate expected node degrees for all nodes based on degree distribution
    parameters. The degree distribution is drawn from scipy's random
    discrete distributions: poisson, uniform (randint), powerlaw (zipf).
    The parameter's given are expected to match those required 
    by the numpy RNG for the chosen distribution.

    Then uses a poisson distribution to draw the number of edges for the graph.

    These edges are then connected according to the connectivity block matrix 
    using the degree corrected SBM (Karrer & Newman, 2010).

    Finally weight distribution parameters are used to draw weights from the
    desired class of distributions: general normal (gennorm), uniform, 
    pareto, log-normal (lognorm), or gamma.

    kwargs: currently supports additional edge attributes
        other_edge_block_attributes: default []
            requires a list of dictionarys with 
            {'distribution_param_matrix', 'key', 
            'distribution', 'distribution_type'}

            distribution_param_matrix - params for each i.j bundle
            key - key to be assigned to the edge attribute
            distribution - name of distribution
            distribution_type - continuous/discrete

    """

    num_groups = len(relative_group_sizes)
    group_sizes = np.array(list(map(int, N * relative_group_sizes 
                                            / np.sum(relative_group_sizes))))
    N = np.sum(group_sizes)

    expected_node_indegrees_by_group, expected_node_outdegrees_by_group = \
                        calculate_expected_degrees(group_sizes, 
                                        degree_distribution_parameter_vector, 
                                        degree_distribution, 
                                        correlated_inout_degree)

    in_connection_prob_by_group = \
        [ calculate_node_connection_probabilities(group_indegress)
        for group_indegress in expected_node_indegrees_by_group ]
    out_connection_prob_by_group = \
        [ calculate_node_connection_probabilities(group_outdegrees)
        for group_outdegrees in expected_node_outdegrees_by_group ]

    node_membership_dictionary = { np.sum(group_sizes[:group]) + node : 
                                    group for group in range(num_groups) 
                                    for node in range(group_sizes[group]) }
    community_to_nodelist_dictionary = { group : [ node for node in range(N) 
        if node_membership_dictionary[node] == group ] 
        for group in range(num_groups) }
    graph = nx.DiGraph()
    graph.add_nodes_from(node_membership_dictionary.keys())
    nx.set_node_attributes(graph, 'community', node_membership_dictionary)

    # Add connections
    edge_bundles_by_group = {}
    for source_group in range(num_groups):
        for target_group in range(num_groups):
            edge_bundles_by_group[(source_group, target_group)] = \
                connect_edge_bundle(graph,
                group_sizes[source_group], group_sizes[target_group],
                community_to_nodelist_dictionary[source_group], 
                community_to_nodelist_dictionary[target_group],
                connectivity_block_matrix[source_group, target_group],
                out_connection_prob_by_group[source_group], 
                in_connection_prob_by_group[target_group])

    # Add weights
    for bundle, edges in edge_bundles_by_group.items():
        add_connection_weights(graph, edges, 
            weight_distribution_parameter_matrix[bundle[0], bundle[1]],
            weight_distribution, negative_weight_fraction_matrix[bundle[0], 
                                                                bundle[1]])

    # Add other attributes
    for edge_block_attributes in other_edge_block_attributes:
        add_edge_attributes(graph, edge_bundles_by_group, 
                            edge_block_attributes)

    if not self_loops:
        remove_self_loops(graph)

    # Relabel all the nodes
    mapping = { old_label: new_label 
            for old_label, new_label in enumerate(np.random.permutation(N)) }
    graph = nx.relabel_nodes(graph, mapping, copy=True)

    return graph

def weighted_directed_stochastic_block_model_asarray(**kwargs):

    return np.asarray(nx.to_numpy_matrix(\
                        weighted_directed_stochastic_block_model(**kwargs)))

# def undirected_stochastic_block_model(N, relative_group_sizes,
#     connectivity_block_matrix, degree_distribution_parameter_vector,
#     self_loops=False, degree_distribution="poisson"):

# def undirected_stochastic_block_model_asarray(**kwargs):

#     return np.asarray(nx.to_numpy_matrix(\
#                         undirected_stochastic_block_model(**kwargs)))

if __name__ == '__main__':
    """
    testing
    """

    import matplotlib.pyplot as plt

    # plt.hist(generate_continuous_distribution(1000, "lognorm", (1.0,1,1)))
    # plt.show()

    N=100
    relative_group_sizes = np.array([0.5, 0.5])
    connectivity_block_matrix = np.array([[0.2, 0.2], [0.2, 0.2]])
    weight_distribution_parameter_matrix = np.array([[(1,.1), (1,.1)],
                                                    [(1,.1), (1,.1)]])
    degree_distribution_parameter_vector = np.array([(1.9,1.5), (1.9,1.5)])
    weight_distribution="uniform"
    degree_distribution="zipf"
    negative_weight_fraction_matrix=np.zeros((2,2))
    correlated_inout_degree=True
    test_graph = weighted_directed_stochastic_block_model(
        N, relative_group_sizes,
        connectivity_block_matrix,
        weight_distribution_parameter_matrix,
        degree_distribution_parameter_vector,
        negative_weight_fraction_matrix,
        weight_distribution,
        degree_distribution,
        correlated_inout_degree)
    nx.write_gexf(test_graph, 'test.gexf')
    # cake = np.asarray(nx.to_numpy_matrix(test_graph))
    cake = nx.fast_gnp_random_graph(100, 0.2, directed=True)
    cc1 = list(cake.in_degree().values())
    # cc2 = list(cake.out_degree().values())
    xx1 = list(test_graph.in_degree().values())
    xx2 = list(test_graph.out_degree().values())
    bins = np.linspace(0,120,41)
    plt.hist(cc1, bins=bins, alpha=0.5, color='r')
    # plt.hist(cc2, bins=bins,alpha=0.5, color='r')
    plt.hist(xx1, bins=bins,alpha=0.5, color='b')
    # plt.hist(xx2, bins=bins,alpha=0.5, color='b')
    plt.show()