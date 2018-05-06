import numpy as np


def unweighted_bipartite_connector_graph(num_origin_nodes, num_sink_nodes,
                                         out_degree, random_state,
                                         dtype=np.uint64):
    """
    Generates a numpy edge array as Ex2 dtype array. All edges are
    directed from origin nodes to sink nodes. Origin is [0] while sink is [1].
    Will randomly connect source nodes to destination nodes so that the out
    degree of the source nodes is preserved.

    The connector graph connects nodes from one graph to another. Origin nodes
    are labelled from 0 to num_origin_nodes - 1. Sink nodes are labelled from
    0 to num_sink_nodes - 1.

    # of edges is out_degree * num_origin_nodes.

    :param num_origin_nodes: The number of origin nodes
    :param num_sink_nodes: The number of sink nodes
    :param out_degree: the out degree of the source nodes
    :param random_state: a numpy RandomState
    :param dtype: the type of numpy array, defaults to np.uint64
    :return: an Ex2 dtype array
    """

    edge_array = np.zeros((num_origin_nodes * out_degree, 2), dtype=dtype)

    # Fill origin
    origin_view = edge_array[:, 0]
    edge = 0
    for node in range(num_origin_nodes):
        for i in range(out_degree):
            origin_view[edge] = node
            edge += 1

    # Fill sink
    sink_view = edge_array[:, 1]
    sink_view[:] = random_state.choice(num_sink_nodes, size=len(sink_view))

    return edge_array


if __name__ == "__main__":
    pass
