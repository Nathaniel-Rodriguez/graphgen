import networkx as nx
import numpy as np
import itertools


def random_product_without_replacement(*args, **kwargs):
    """
    Pulls random products of multiple lists without replacement.
    Works well for sparse calls that are many times less than
    the total number of combinations.
    """

    size = kwargs.get('size', 1)
    random_state = kwargs['random_state']
    used = set()
    pools = list(map(tuple, args))
    
    # Check that size doesn't exceed the number of available draws
    max_product = 1
    for pool in pools:
        max_product *= len(pool)
        
    if size > max_product:
        print('Warning: size exceeds max product!', ' Max', max_product)
        return None
    
    # If size on the same order of magnitude as the number of 
    # products just build a list of the options and pull from that
    elif size >= (max_product / 10):
        products = list(itertools.product(*pools))
        random_state.shuffle(products)
        return products[:size]
    
    # if size is much less than the number of products generate rv's on the fly
    else:
        rvs = []
        for _ in range(size):
            # rv = tuple(random.choice(pool) for pool in pools)
            rv = tuple(random_state.choice(pool) for pool in pools)
            while rv in used:
                # rv = tuple(random.choice(pool) for pool in pools)
                rv = tuple(random_state.choice(pool) for pool in pools)

            rvs.append(rv)
            used.add(rv)

    return rvs


def build_node2membership_translator(num_of_levels, communities_per_level, base_com_size):
    """
    Translate: nodes -> membership (given level)
    """
    membership_struct = {l: {membership * base_com_size
                             * communities_per_level**l + j: membership
                             for membership in range(int(communities_per_level**num_of_levels
                                                         / communities_per_level**l))
                             for j in range(base_com_size * communities_per_level**l)}
                         for l in range(num_of_levels)}

    return membership_struct


def build_membership2node_translator(node2membership_translator):

    # Translate: membership -> nodes (given level)
    keyLevel_valMembershipDict = {}
    for l, keyNode_valMembership in node2membership_translator.items():
        keyMembership_valNodes = {}
        for node, membership in keyNode_valMembership.items():
            if membership in keyMembership_valNodes:
                keyMembership_valNodes[membership].append(node)
            else:
                keyMembership_valNodes[membership] = [node]

        keyLevel_valMembershipDict[l] = keyMembership_valNodes

    return keyLevel_valMembershipDict


def assign_community_memberships(graph, num_of_levels, node2membership_translator):

    # Assign memberships
    for l in range(num_of_levels):
        nx.set_node_attributes(graph, name=l, values=node2membership_translator[l])


def connect_base_layer(graph, membership2node_translator):
    """
    Currently builds full list of permutations, may switch to generator
    Use 4 loop through combinations to do a memory friendly use of the generator w/ add edges - for larger graphs
    """
    
    # Fully connect lowest layer
    for nodes in membership2node_translator[0].values():
        graph.add_edges_from(list(itertools.permutations(nodes, 2)))


def connect_upper_layers(graph, num_of_levels, communities_per_level,
                         attachment_probability, connectivity_scaling,
                         membership2node_translator, random_state):
    """
    :param graph: current graph
    :param num_of_levels: number of levels in hierarchy
    :param communities_per_level: number of communities for each level
    :param attachment_probability: probability of attaching nodes
    :param connectivity_scaling: scaling of node attachment probability between levels
    :param membership2node_translator: dictionary for node community membership
    :param random_state: numpy RandomState
    :return: none
    """

    # Probabilistically connect other layers
    for l in range(1, num_of_levels):

        # Construct block-set list
        block_sets = [[i+j for j in range(communities_per_level)]
                      for i in range(0, len(membership2node_translator[l]),
                                     communities_per_level)]

        for block_set in block_sets:
            for block1, block2 in itertools.permutations(block_set, 2):

                # Use poisson to pick number of edges
                size_of_group1 = len(membership2node_translator[l][block1])
                size_of_group2 = len(membership2node_translator[l][block2])
                num_of_expected_events = connectivity_scaling * attachment_probability ** l * \
                    ((size_of_group1 + size_of_group2) / 2)**2
                num_edges_1to2, num_edges_2to1 = random_state.poisson(num_of_expected_events, size=2)

                # Make sure graph is connected and edges doesn't exceed max possible connections
                if num_edges_1to2 < 1:
                    num_edges_1to2 = 1
                elif num_edges_1to2 > size_of_group1 * size_of_group2:
                    num_edges_1to2 = size_of_group1 * size_of_group2
                if num_edges_2to1 < 1:
                    num_edges_2to1 = 1
                elif num_edges_2to1 > size_of_group1 * size_of_group2:
                    num_edges_2to1 = size_of_group1 * size_of_group2

                random_edges_1to2 = random_product_without_replacement(membership2node_translator[l][block1],
                                                                       membership2node_translator[l][block2],
                                                                       size=num_edges_1to2,
                                                                       random_state=random_state)
                random_edges_2to1 = random_product_without_replacement(membership2node_translator[l][block2],
                                                                       membership2node_translator[l][block1],
                                                                       size=num_edges_2to1,
                                                                       random_state=random_state)

                graph.add_edges_from(random_edges_1to2)
                graph.add_edges_from(random_edges_2to1)


def unweighted_directed_hmn_as_nx(num_of_levels, communities_per_level,
                                  base_com_size, attachment_probability,
                                  connectivity_scaling, random_state):
    """
    Builds a hierarchical modular network using the aglorithm from:
    Moretti, P., & Munoz, M. A. (2013). Griffiths phases and the stretching of
    criticality in brain networks. Nature Communications, 4, 2521.
    https://doi.org/10.1038/ncomms3521

    Uses networkx and makes directed graphs.

    :param num_of_levels: the number of levels in the hierarchy
    :param communities_per_level: the number of communities for each level
    :param base_com_size: the size of the lowest communities
    :param attachment_probability: probability of attaching nodes
    :param connectivity_scaling: scaling of node attachment probability between levels
    :param random_state: a numpy RandomState
    :return: returns a networkx graph
    """

    # Generate initial node set
    num_nodes = base_com_size * communities_per_level ** num_of_levels
    graph = nx.DiGraph()
    graph.add_nodes_from(range(num_nodes))

    node2membership_translator = build_node2membership_translator(num_of_levels,
                                                                  communities_per_level,
                                                                  base_com_size)
    membership2node_translator = build_membership2node_translator(node2membership_translator)
    assign_community_memberships(graph, num_of_levels, node2membership_translator)
    connect_base_layer(graph, membership2node_translator)
    connect_upper_layers(graph, num_of_levels, communities_per_level, attachment_probability,
        connectivity_scaling, membership2node_translator, random_state)

    return graph


def unweighted_directed_hmn_as_asarray(**kwargs):

    return np.asarray(nx.to_numpy_matrix(unweighted_directed_hmn_as_nx(**kwargs)))


def unweighted_directed_hmn_as_edge_list(dtype=np.uint64, **kwargs):
    """
    :param dtype: numpy type of edge list, defaults: uint64
    :param kwargs: arguments for unweighted_directed_hmn_as_nx
    :return: (numpy array Ex2, number of nodes)
    """

    graph = unweighted_directed_hmn_as_nx(**kwargs)
    edge_list = np.zeros((graph.number_of_edges(), 2), dtype=dtype)
    for i, edge in enumerate(graph.edges()):
        edge_list[i, 0] = edge[0]
        edge_list[i, 1] = edge[1]

    return edge_list, graph.number_of_nodes()


if __name__ == '__main__':
    """
    """

    # Parameters
    alpha = 2
    b = 2
    s = 10
    M_o = 2
    p = 1 / 4.

    N = M_o * b ** s
    print(N)

    # Generate initial node set
    graph = unweighted_directed_hmn(s, b, M_o, p, alpha,
                                    np.random.RandomState(3423))
    print(nx.info(graph))

    nx.write_gexf(graph,"test.gexf")